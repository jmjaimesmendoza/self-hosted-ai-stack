-- Row-level security setup for the Telegram bot.
--
-- Creates a SELECT-only role `speech_sql_user` whose visibility is limited to
-- one organization per query, selected via:  set_config('app.org_id', <id>, true)
-- (bot.py does this inside every read-only transaction).
--
-- Run as the DB user that owns the tables (the one Prisma migrates with):
--   psql "$ADMIN_DATABASE_URL" -v bot_password='<strong password>' -f rls.sql
-- If speech_sql_user already exists its password is left unchanged (the
-- bot_password value is ignored; pass anything). The role must not own the
-- application tables — table owners bypass RLS.
--
-- Idempotent — re-run after every Prisma migration so new tables get policies
-- (new tables are invisible to the bot until you do; safe by default).
--
-- How policies are generated:
--   1. Tables with an organization_id column get a direct policy.
--   2. Other tables get an EXISTS policy hopping one FK to an already-scoped
--      parent; Postgres applies RLS recursively, so chains like
--      details -> events -> persons -> org resolve automatically.
--   3. Leftovers with no org path are denied, except a whitelist of global
--      reference tables (countries, breeds, ...).
-- The backend is unaffected: it connects as the table owner, which bypasses
-- non-FORCE RLS, and all policies are TO speech_sql_user only.

SELECT format('CREATE ROLE speech_sql_user LOGIN PASSWORD %L', :'bot_password')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'speech_sql_user') \gexec

-- The role may pre-exist: make sure it can't sidestep RLS or DDL its way out
ALTER ROLE speech_sql_user LOGIN NOSUPERUSER NOBYPASSRLS NOCREATEDB NOCREATEROLE;

GRANT USAGE ON SCHEMA public TO speech_sql_user;

DO $$
DECLARE
  r record;
  fk record;
  cond text;
  policied text[] := ARRAY[]::text[];
  denied text[] := ARRAY[]::text[];
  -- global reference tables: readable by every org, no RLS needed
  allowed_global text[] := ARRAY[
    'countries', 'cattle_breeds', 'variables', 'farm_variables',
    'variable_template_keys', 'functions', 'permissions', 'tutorials',
    'spatial_ref_sys'
  ];
  -- sensitive or org-less tables the bot must never read
  always_deny text[] := ARRAY[
    'user_tokens', 'information_requests', '_prisma_migrations',
    'spatial_ref_sys_dummy'
  ];
  progress boolean;
BEGIN
  -- Grant SELECT per table (skip tables we don't own, e.g. postgis internals)
  FOR r IN
    SELECT c.relname AS tbl FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind = 'r'
  LOOP
    BEGIN
      -- reset to SELECT-only, dropping any write grants from the role's past
      EXECUTE format('REVOKE ALL ON public.%I FROM speech_sql_user', r.tbl);
      EXECUTE format('GRANT SELECT ON public.%I TO speech_sql_user', r.tbl);
    EXCEPTION WHEN insufficient_privilege THEN
      RAISE NOTICE 'skipped grant on % (not owner)', r.tbl;
    END;
  END LOOP;

  -- organizations has no organization_id column: its own id is the org
  EXECUTE 'ALTER TABLE public.organizations ENABLE ROW LEVEL SECURITY';
  EXECUTE 'DROP POLICY IF EXISTS bot_org ON public.organizations';
  EXECUTE $q$CREATE POLICY bot_org ON public.organizations FOR SELECT TO speech_sql_user
             USING (id = current_setting('app.org_id', true))$q$;
  policied := array_append(policied, 'organizations');

  -- Pass 1: direct organization_id column
  FOR r IN
    SELECT c.relname AS tbl FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind = 'r'
      AND c.relname <> ALL (always_deny)
      AND EXISTS (SELECT 1 FROM pg_attribute a
                  WHERE a.attrelid = c.oid AND a.attname = 'organization_id'
                    AND a.attnum > 0 AND NOT a.attisdropped)
  LOOP
    EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', r.tbl);
    EXECUTE format('DROP POLICY IF EXISTS bot_org ON public.%I', r.tbl);
    EXECUTE format(
      $q$CREATE POLICY bot_org ON public.%I FOR SELECT TO speech_sql_user
         USING (organization_id = current_setting('app.org_id', true))$q$, r.tbl);
    policied := policied || r.tbl;
  END LOOP;

  -- Pass 2..n: hop one FK to an already-scoped parent, until fixpoint
  LOOP
    progress := false;
    FOR r IN
      SELECT c.relname AS tbl, c.oid FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
      WHERE n.nspname = 'public' AND c.relkind = 'r'
        AND c.relname <> ALL (policied)
        AND c.relname <> ALL (allowed_global)
        AND c.relname <> ALL (always_deny)
    LOOP
      SELECT con.conrelid, con.confrelid, pc.relname AS parent,
             con.conkey, con.confkey
        INTO fk
      FROM pg_constraint con
      JOIN pg_class pc ON pc.oid = con.confrelid
      WHERE con.conrelid = r.oid AND con.contype = 'f'
        AND con.confrelid <> con.conrelid
        AND pc.relname = ANY (policied)
      -- prefer NOT NULL FKs: nullable-FK rows are hidden (org unknown = unsafe)
      ORDER BY (SELECT bool_and(a.attnotnull) FROM pg_attribute a
                WHERE a.attrelid = con.conrelid AND a.attnum = ANY (con.conkey)) DESC
      LIMIT 1;

      IF FOUND THEN
        SELECT string_agg(format('p.%I = %I.%I', fa.attname, r.tbl, ca.attname),
                          ' AND ')
          INTO cond
        FROM (SELECT ck.attnum AS c_attnum, fkk.attnum AS p_attnum
              FROM unnest(fk.conkey) WITH ORDINALITY AS ck(attnum, ord)
              JOIN unnest(fk.confkey) WITH ORDINALITY AS fkk(attnum, ord)
                ON ck.ord = fkk.ord) m
        JOIN pg_attribute ca ON ca.attrelid = fk.conrelid AND ca.attnum = m.c_attnum
        JOIN pg_attribute fa ON fa.attrelid = fk.confrelid AND fa.attnum = m.p_attnum;

        EXECUTE format('ALTER TABLE public.%I ENABLE ROW LEVEL SECURITY', r.tbl);
        EXECUTE format('DROP POLICY IF EXISTS bot_org ON public.%I', r.tbl);
        EXECUTE format(
          'CREATE POLICY bot_org ON public.%I FOR SELECT TO speech_sql_user
           USING (EXISTS (SELECT 1 FROM public.%I p WHERE %s))',
          r.tbl, fk.parent, cond);
        policied := policied || r.tbl;
        progress := true;
      END IF;
    END LOOP;
    EXIT WHEN NOT progress;
  END LOOP;

  -- Deny everything without an org path (plus the explicit deny list)
  FOR r IN
    SELECT c.relname AS tbl FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE n.nspname = 'public' AND c.relkind = 'r'
      AND c.relname <> ALL (policied)
      AND c.relname <> ALL (allowed_global)
  LOOP
    BEGIN
      EXECUTE format('REVOKE SELECT ON public.%I FROM speech_sql_user', r.tbl);
      denied := denied || r.tbl;
    EXCEPTION WHEN OTHERS THEN NULL;
    END;
  END LOOP;

  RAISE NOTICE 'org-scoped tables: %', array_to_string(policied, ', ');
  RAISE NOTICE 'denied tables: %', array_to_string(denied, ', ');
END $$;

-- The bot must never see password hashes or reset tokens: replace the
-- table-level grant on users with a column grant. (SELECT * on users will
-- fail for the bot; per-column selects work.)
REVOKE SELECT ON public.users FROM speech_sql_user;
GRANT SELECT (id, name, email, organization_id, profile_id, locale,
              is_super_admin, created_at, updated_at, is_deleted)
  ON public.users TO speech_sql_user;

-- Login runs before an org is known, so it can't go through RLS: a
-- SECURITY DEFINER function returns exactly the one row login needs.
-- locale is resolved here: user override, else the org default.
DROP FUNCTION IF EXISTS public.bot_login(text);
CREATE FUNCTION public.bot_login(p_email text)
RETURNS TABLE (id text, name text, password text, organization_id text, locale text)
LANGUAGE sql STABLE SECURITY DEFINER
SET search_path = public
AS $fn$
  SELECT u.id, u.name, u.password, u.organization_id,
         COALESCE(u.locale, o.locale)::text
  FROM users u
  JOIN organizations o ON o.id = u.organization_id
  WHERE u.email = p_email AND u.is_deleted = false AND o.is_active = true
$fn$;

REVOKE ALL ON FUNCTION public.bot_login(text) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.bot_login(text) TO speech_sql_user;
