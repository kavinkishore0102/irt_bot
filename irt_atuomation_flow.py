@task
class IrtAutomationFlow:
    """
    IrtAutomationFlow task class handles automation categories for the IRT team.

    This class manages:
    - Input validation for each automation category
    - ArangoDB queries and updates

    Attributes:
        config : Configuration settings (None by default)
    """

    import os
    import re
    import time
    import asyncio
    import json
    from typing import Optional
    from datetime import datetime, timezone
    from arango import ArangoClient
    from arango.exceptions import AQLQueryExecuteError

    # ── ArangoDB config (read from env.sh) ───────────────────────────────
    ARANGO_SERVER   = "http://10.0.1.121:8529"
    ADMIN_DB = "thickstatAdmin"
    ATHENA_DB = "thickstatAthena"
    INFRA_DB = "infrastructure"
    ARANGO_USERNAME = "application"
    ARANGO_PASSWORD = "v2uI1HE5xh5j81"

    # ── Date validation regex ─────────────────────────────────────────────
    DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    # ── Supported Timezones ───────────────────────────────────────────────
    SUPPORTED_TIMEZONES = [
        "ART", "AEST", "ACST", "AWST", "BRT", "AMT", "ACT", "ICT", "NST", "AST",
        "EST", "CST", "MST", "PST", "EET", "CET", "WET", "IST", "WIB", "WITA",
        "WIT", "IRST", "JST", "EAT", "MYT", "NZDT", "NZST", "PHT", "MSK", "SAMT",
        "YEKT", "OMST", "KRAT", "IRKT", "YAKT", "VLAT", "MAGT", "PETT", "SGT",
        "KST", "TRT", "GST", "GMT", "BST", "PT", "MT", "HST", "AKDT", "PDT",
        "MDT", "CDT", "EDT", "AKST", "ET"
    ]

    def __init__(self):
        """TBW"""
        self.config        = None
        self.arango_client = None   # ArangoClient instance
        self.db_client     = None   # StandardDatabase instance

    # ═══════════════════════════════════════════════════════════════════════
    # ARANGODB METHODS
    # ═══════════════════════════════════════════════════════════════════════

    async def connect_arango(self, ctx: CSContext):
        """
        Establish connection to the ArangoDB server.

        Args:
            ctx (CSContext): Context object for logging

        Returns:
            ArangoClient instance if successful, Exception otherwise.
        """
        try:
            client = self.ArangoClient(self.ARANGO_SERVER)
            ctx.log.info(f"[ArangoDB] Server connected → {self.ARANGO_SERVER}")
            return client
        except Exception as e:
            ctx.log.error(f"[ArangoDB] Server connection failed → {self.ARANGO_SERVER} → {e}")
            return e

    async def connect_db(self, db_name: str, ctx: CSContext):
        """
        Connect to the specified ArangoDB database.

        Args:
            db_name (str): Name of the database to connect to.
            ctx (CSContext): Context object for logging

        Returns:
            StandardDatabase client if successful, Exception otherwise.
        """
        try:
            db_client = self.arango_client.db(
                db_name,
                self.ARANGO_USERNAME,
                self.ARANGO_PASSWORD
            )
            ctx.log.info(f"[ArangoDB] Database [{db_name}] connected successfully")
            return db_client
        except Exception as e:
            ctx.log.error(f"[ArangoDB] Database [{db_name}] connection failed → {e}")
            return e

    async def execute_query(self, query: str, ctx: CSContext, bind_vars: dict = {}):
        """
        Execute an AQL query with retry logic (3 attempts on connection reset).

        Args:
            query (str)      : AQL query string to execute
            ctx (CSContext)  : Context object for logging
            bind_vars (dict) : Optional dictionary of bind variables

        Returns:
            tuple: (cursor, error) — cursor is iterable result, error is None on success.
        """
        try:
            max_try    = 3
            incr_count = 2
            try_count  = 0
            sleep_time = 3

            while try_count < max_try:
                try:
                    ctx.log.debug(f"[ArangoDB] Executing query → {query}")
                    ctx.log.debug(f"[ArangoDB] Database → [{self.db_client.name}]")

                    if bind_vars:
                        cursor = self.db_client.aql.execute(query, bind_vars=bind_vars)
                    else:
                        cursor = self.db_client.aql.execute(query)
                    return cursor, None

                except self.AQLQueryExecuteError as err:
                    if err.error_code is not None and err.error_code == 1620:
                        ctx.log.error(f"[ArangoDB] Schema validation failed → {err}")
                    else:
                        ctx.log.error(f"[ArangoDB] AQL execution error → {err}")
                    return None, Exception(err.error_message)

                except Exception as err:
                    if "ConnectionResetError" in str(err):
                        ctx.log.error(
                            f"[ArangoDB] Connection reset, retrying in {sleep_time}s: {err}"
                        )
                        self.time.sleep(sleep_time)
                        sleep_time *= incr_count
                        try_count  += 1
                        if try_count == max_try:
                            ctx.log.error(
                                f"[ArangoDB] Tried {max_try} times, still getting connection reset. Exiting."
                            )
                    else:
                        ctx.log.error(f"[ArangoDB] Query execution error → {err}")
                        return None, err

        except Exception as e:
            ctx.log.error(f"[ArangoDB] Error in execute_query: {e}")
            return None, e

    async def get_org_document(self, org_id: str, ctx: CSContext) -> Optional[dict]:
        """
        Fetches the organization document for the given org_id.

        Args:
            org_id (str)     : Organization ID, e.g. "trailtest01"
            ctx (CSContext)  : Context object for logging

        Returns:
            dict — the organization document, or None if not found.
        """
        try:
            query       = f"FOR doc IN organization FILTER doc.orgId == '{org_id}' RETURN doc"
            cursor, err = await self.execute_query(query, ctx)
            
            if err:
                ctx.log.error(f"[ArangoDB] Failed to fetch org document: {err}")
                return None
            results = list(cursor)
            return results[0] if results else None
        except Exception as e:
            ctx.log.error(f"[ArangoDB] get_org_document error: {e}")
            raise e

    async def update_trial_expiry_time(self, org_id: str, new_expiry_time: str, ctx: CSContext) -> bool:
        """
        Updates tierSettings.trialSettings.expiryTime for the given org_id.

        Uses AQL MERGE so only expiryTime is patched — all other sibling fields
        inside trialSettings are preserved.

        Args:
            org_id          (str)       : Organization ID, e.g. "trailtest01"
            new_expiry_time (str)       : ISO 8601 UTC string, e.g. "2026-04-20T00:00:00.000Z"
            ctx             (CSContext) : Context object for logging

        Returns:
            True if at least one document was updated, False otherwise.
        """
        try:
            query = f"""
            FOR doc IN organization
                FILTER doc.orgId == '{org_id}'
                UPDATE doc WITH {{
                    tierSettings: MERGE(doc.tierSettings, {{
                        trialSettings: MERGE(doc.tierSettings.trialSettings, {{
                            expiryTime: '{new_expiry_time}'
                        }})
                    }})
                }} IN organization
                RETURN NEW
            """
            cursor, err = await self.execute_query(query, ctx)
            
            if err:
                ctx.log.error(f"[ArangoDB] Failed to update trial expiry: {err}")
                return False
            results = list(cursor)
            if results:
                ctx.log.info(f"[ArangoDB] Updated expiryTime for org '{org_id}' → {new_expiry_time}")
                return True
            ctx.log.error(f"[ArangoDB] No document updated for org '{org_id}'.")
            return False
        except Exception as e:
            ctx.log.error(f"[ArangoDB] update_trial_expiry_time error: {e}")
            raise e

    async def update_refresh_time(self, org_id: str, timezone: str, refresh_time: list, ctx: CSContext) -> bool:
        """
        Updates the timezone and refreshTime for the given org_id.

        Args:
            org_id       (str)       : Organization ID
            timezone     (str)       : Timezone string, e.g. "AEST"
            refresh_time (list)      : List of refresh time strings, e.g. ["2025-08-26T09:30:00Z"]
            ctx          (CSContext) : Context object for logging

        Returns:
            True if at least one document was updated, False otherwise.
        """
        try:
            refresh_time_json = self.json.dumps(refresh_time)
            query = f"""
            FOR doc IN organization
                FILTER doc.orgId == '{org_id}'
                UPDATE doc WITH {{
                    localeInfo: {{
                        timezone: '{timezone}'
                    }},
                    tierSettings: {{
                        dataRefresh: {{
                            refreshTime: {refresh_time_json}
                        }}
                    }}
                }} IN organization
                RETURN NEW
            """
            cursor, err = await self.execute_query(query, ctx)
            
            if err:
                ctx.log.error(f"[ArangoDB] Failed to update refresh time: {err}")
                return False
            results = list(cursor)
            if results:
                ctx.log.info(f"[ArangoDB] Updated refreshTime for org '{org_id}'")
                return True
            ctx.log.error(f"[ArangoDB] No document updated for org '{org_id}'.")
            return False
        except Exception as e:
            ctx.log.error(f"[ArangoDB] update_refresh_time error: {e}")
            raise e

    async def change_user_email_impl(self, role: str, old_email: str, new_email: str, user_id: str, dataset_id: str, org_id: str, ctx: CSContext) -> bool:
        """
        Updates a user's email across collections.
        If role is 'admin', it also updates dataset_v1 and organization in the Admin DB.
        """
        try:
            timestamp = self.datetime.now(self.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')

            # --- 1. Update ADMIN DB collections ('user' and 'authentication') ---
            # self.db_client is already connected to ADMIN_DB by default in run()

            user_query = """
                FOR doc IN user 
                FILTER doc.email == @old_email 
                UPDATE doc WITH {
                    email: @new_email, 
                    updatedAt: @updatedAt
                } IN user
            """

            auth_query = """
                FOR doc IN authentication 
                FILTER doc.email == @old_email 
                UPDATE doc WITH {
                    email: @new_email, 
                    updatedAt: @updatedAt
                } IN authentication
            """

            base_bind_vars = {
                "old_email": old_email,
                "new_email": new_email,
                "updatedAt": timestamp
            }

            _, err1 = await self.execute_query(user_query, ctx, bind_vars=base_bind_vars)
            _, err2 = await self.execute_query(auth_query, ctx, bind_vars=base_bind_vars)

            if err1 or err2:
                ctx.log.error(f"[AdminEmailChanges] Failed to update user/auth collections: err1={err1}, err2={err2}")
                return False

            ctx.log.info(f"[AdminEmailChanges] Successfully updated 'user' and 'authentication' for {old_email}")

            # --- 2. Update additional collections if role is 'admin' ---
            if role.lower() == 'admin':
                if not all([user_id, dataset_id, org_id]):
                    raise ValueError("Admin role requires 'user_id', 'dataset_id', and 'org_id' to be provided.")

                # a) Update dataset_v1 in ATHENA DB
                athena_db_client = await self.connect_db(self.ATHENA_DB, ctx)
                if isinstance(athena_db_client, Exception):
                    ctx.log.error(f"[AdminEmailChanges] Failed to connect to Athena DB: {athena_db_client}")
                    return False
                
                original_db_client = self.db_client
                self.db_client     = athena_db_client

                dataset_query = """
                    FOR doc IN dataset_v1
                    FILTER doc._key == @datasetId
                    UPDATE doc WITH {
                        user_id: @userId,
                        updated_by: @newEmail,
                        updated_at: @updatedAt
                    } IN dataset_v1
                """

                dataset_bind_vars = {
                    "datasetId": dataset_id,
                    "userId": user_id,
                    "newEmail": new_email,
                    "updatedAt": timestamp
                }

                _, err3 = await self.execute_query(dataset_query, ctx, bind_vars=dataset_bind_vars)
                
                # b) Restore ADMIN DB client to update organization
                self.db_client = original_db_client

                org_query = """
                    FOR doc IN organization
                    FILTER doc._key == @orgId
                    UPDATE doc WITH {
                        updatedAt: @updatedAt,
                        userId: @userId
                    } IN organization
                """

                org_bind_vars = {
                    "orgId": org_id,
                    "userId": user_id,
                    "updatedAt": timestamp
                }

                _, err4 = await self.execute_query(org_query, ctx, bind_vars=org_bind_vars)

                if err3 or err4:
                    ctx.log.error(f"[AdminEmailChanges] Failed to update dataset/org for admin: err3={err3}, err4={err4}")
                    return False
                
                ctx.log.info(f"[AdminEmailChanges] Successfully updated 'dataset_v1' and 'organization' for admin {old_email}")

            return True

        except Exception as e:
            ctx.log.error(f"[AdminEmailChanges] change_user_email_impl error: {e}")
            raise e

    async def enable_athena_threads_impl(self, org_id: str, dataset_id: str, template_org_id: str, ctx: CSContext) -> bool:
        """
        Enables Athena threads for a dataset and updates proactive insights from a template.
        """
        try:
            # 1. Fetch Organization Foresight (from Admin DB, which is default)
            foresight_query = """
                FOR doc IN organization
                FILTER doc.orgId == @orgId
                RETURN doc.foresight
            """
            cursor, err = await self.execute_query(foresight_query, ctx, bind_vars={"orgId": org_id})
            if err:
                ctx.log.error(f"[EnableAthenaThreads] Failed to fetch foresight for org {org_id}: {err}")
                return False
                
            results = list(cursor)
            foresight = results[0] if results else None

            if foresight != "fishbowl":
                ctx.log.info(f"[EnableAthenaThreads] Athena threads skipped — foresight is '{foresight}' for org {org_id}")
                return True

            # Switch to Athena DB for subsequent queries
            athena_db_client = await self.connect_db(self.ATHENA_DB, ctx)
            if isinstance(athena_db_client, Exception):
                ctx.log.error(f"[EnableAthenaThreads] Failed to connect to Athena DB: {athena_db_client}")
                return False

            original_db_client = self.db_client
            self.db_client = athena_db_client

            # 2. Update dataset_v1
            update_dataset_query = """
                FOR doc IN dataset_v1
                FILTER doc._key == @datasetId
                UPDATE doc WITH {
                    athena_threads: true,
                    updated_at: @updatedAt
                } IN dataset_v1
            """

            updated_at = self.datetime.now(self.timezone.utc).isoformat(timespec='microseconds').replace('+00:00', 'Z')
            dataset_bind_vars = {
                "datasetId": dataset_id,
                "updatedAt": updated_at
            }

            _, err_ds = await self.execute_query(update_dataset_query, ctx, bind_vars=dataset_bind_vars)
            if err_ds:
                self.db_client = original_db_client
                ctx.log.error(f"[EnableAthenaThreads] Failed to enable athena threads for dataset {dataset_id}: {err_ds}")
                return False
                
            ctx.log.info(f"[EnableAthenaThreads] Athena threads enabled for dataset {dataset_id}")

            # 3. Update proActiveInsightComponents dynamically from templates
            # update_insights_query = """
            #     LET templateMapping = MERGE(
            #         (FOR t IN template
            #             FILTER t.collection_type == 'proactive'
            #             FILTER t.orgID == @templateOrgId
            #             FOR pMap IN t.proactive.proActiveMap
            #                 FOR comp IN pMap.proActiveInsightsComponents
            #                     FILTER comp.followup.question != null AND comp.followup.question != ""
            #                     RETURN {
            #                         [comp.followup.question]: {
            #                             mode: comp.followup.mode,
            #                             semantics: comp.followup.semantics
            #                         }
            #                     }
            #         )
            #     )
            #     FOR doc IN proActiveInsightComponents
            #         FILTER doc.orgID == @targetOrgId
            #         LET matchedTemplate = templateMapping[doc.followup.question]
            #         FILTER matchedTemplate != null AND matchedTemplate.semantics != null
            #         UPDATE doc WITH {
            #             followup: MERGE(doc.followup, {
            #                 mode: matchedTemplate.mode,
            #                 semantics: matchedTemplate.semantics
            #             })
            #         } IN proActiveInsightComponents
            # """

            update_insights_query = """
                    LET templateMapping = MERGE(
                        (FOR t IN template
                            FILTER t.collection_type == 'proactive'
                            FOR pMap IN (t.proactive.proActiveMap || [])
                                FOR comp IN (pMap.proActiveInsightsComponents || [])
                                    FILTER comp.orgID == @templateOrgId
                                    FILTER comp.followup.question != null AND comp.followup.question != ""
                                    RETURN {
                                        [comp.followup.question]: {
                                            mode: comp.followup.mode,
                                            semantics: comp.followup.semantics
                                        }
                                    }
                        )
                    )
                    FOR doc IN proActiveInsightComponents
                        FILTER doc.orgID == @targetOrgId
                        LET matchedTemplate = templateMapping[doc.followup.question]
                        FILTER matchedTemplate != null AND matchedTemplate.semantics != null
                        UPDATE doc WITH {
                            followup: MERGE(doc.followup, {
                                mode: matchedTemplate.mode,
                                semantics: matchedTemplate.semantics
                            })
                        } IN proActiveInsightComponents
                """

            insight_bind_vars = {
                "targetOrgId": org_id,
                "templateOrgId": template_org_id,
            }

            _, err_ins = await self.execute_query(update_insights_query, ctx, bind_vars=insight_bind_vars)
            
            # Restore db client
            self.db_client = original_db_client

            if err_ins:
                ctx.log.error(f"[EnableAthenaThreads] Failed to update mode and semantics in Insights for {org_id}: {err_ins}")
                return False
                
            ctx.log.info(f"[EnableAthenaThreads] Successfully updated mode and semantics in Insights for {org_id}")
            return True

        except Exception as e:
            ctx.log.error(f"[EnableAthenaThreads] enable_athena_threads_impl error: {e}")
            raise e

    aasync def get_knowledge_base_entity_count(self, tenant_id: str, ctx: CSContext) -> int:
        """
        Fetches the exact count of entity points for a specific tenant in the Qdrant knowledge base.

        Args:
            tenant_id (str)       : The ID of the tenant/collection (e.g., "691e74ff-sDO_SbiDm")
            ctx       (CSContext) : Context object for logging

        Returns:
            int: The exact count of entities, or -1 if the request fails.
        """
        try:
            # Note: Hardcoded for now. In production, consider moving to self.config
            qdrant_url = "https://fe2dfbd0-87d6-4e7b-b5af-c8eaa76d3e93.us-east4-0.gcp.cloud.qdrant.io:6333"
            api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.xkh-y69TjdQGXenawijf6w6tRGRSHFa8T9nnYMN_dHQ"
            
            url = f"{qdrant_url}/collections/knowledge_base/points/count"

            payload = {
                "filter": {
                    "must": [
                        {
                            "key": "content_type",
                            "match": {
                                "value": "entity"
                            }
                        },
                        {
                            "key": "tenant_id",
                            "match": {
                                "value": tenant_id
                            }
                        }
                    ]
                },
                "exact": True
            }

            headers = {
                "api-key": api_key,
                "Content-Type": "application/json",
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload, timeout=30) as resp:
                    
                    if resp.status != 200:
                        err_text = await resp.text()
                        ctx.log.error(f"[GetKBEntityCount] Failed to fetch count for tenant '{tenant_id}'. Status: {resp.status}, Error: {err_text}")
                        return -1
                    
                    data = await resp.json()
                    count = data.get("result", {}).get("count", 0)
                    
                    ctx.log.info(f"[GetKBEntityCount] Successfully retrieved entity count ({count}) for tenant '{tenant_id}'")
                    return count

        except Exception as e:
            ctx.log.error(f"[GetKBEntityCount] get_knowledge_base_entity_count error for tenant '{tenant_id}': {e}")
            raise e

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATION METHODS
    # ═══════════════════════════════════════════════════════════════════════

    async def validate_timezone(self, timezone_str: str, ctx: CSContext) -> None:
        """
        Validates that the given timezone is in the supported list.

        Args:
            timezone_str : Timezone abbreviation to validate
            ctx          : CSContext — for logging

        Raises:
            ValueError: With a descriptive message if validation fails.
        """
        try:
            if timezone_str not in self.SUPPORTED_TIMEZONES:
                ctx.log.error(
                    f"[Validation] Invalid timezone: '{timezone_str}'. Supported timezones: {self.SUPPORTED_TIMEZONES}"
                )
                raise ValueError(f"Invalid timezone: '{timezone_str}'. Must be one of the supported timezones.")
        except Exception as e:
            ctx.log.error(f"[Validation] validate_timezone error: {e}")
            raise e

    async def convert_refresh_time_to_utc(self, refresh_time_str: str, timezone_str: str, ctx: CSContext) -> str:
        """
        Converts a local time string in a specific timezone to a UTC ISO string.

        Args:
            refresh_time_str : Local time string, e.g. "2025-08-26T09:30"
            timezone_str     : Timezone abbreviation, e.g. "AEST"
            ctx              : CSContext — for logging

        Returns:
            UTC ISO string, e.g. "2025-08-25T23:30:00.000Z"
        """
        try:
            from dateutil import tz
            from dateutil.parser import parse

            offsets = {
                "ART": -3, "AEST": 10, "ACST": 9.5, "AWST": 8, "BRT": -3, "AMT": 4, "ACT": -5, "ICT": 7, "NST": -3.5, "AST": -4,
                "EST": -5, "CST": -6, "MST": -7, "PST": -8, "EET": 2, "CET": 1, "WET": 0, "IST": 5.5, "WIB": 7, "WITA": 8,
                "WIT": 9, "IRST": 3.5, "JST": 9, "EAT": 3, "MYT": 8, "NZDT": 13, "NZST": 12, "PHT": 8, "MSK": 3, "SAMT": 4,
                "YEKT": 5, "OMST": 6, "KRAT": 7, "IRKT": 8, "YAKT": 9, "VLAT": 10, "MAGT": 11, "PETT": 12, "SGT": 8,
                "KST": 9, "TRT": 3, "GST": 4, "GMT": 0, "BST": 1, "PT": -8, "MT": -7, "HST": -10, "AKDT": -8, "PDT": -7,
                "MDT": -6, "CDT": -5, "EDT": -4, "AKST": -9, "ET": -5
            }

            offset_hours = offsets.get(timezone_str, 0)
            tz_info = tz.tzoffset(timezone_str, offset_hours * 3600)
            
            # Parse the time string
            dt = parse(refresh_time_str)
            
            # If the string doesn't provide offset info, treat it as the target timezone
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz_info)
            else:
                # If it already has tzinfo but not matching, adjust it
                dt = dt.astimezone(tz_info)
            
            # Convert to UTC
            dt_utc = dt.astimezone(self.timezone.utc)
            return dt_utc.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        except Exception as e:
            ctx.log.error(f"[Validation] convert_refresh_time_to_utc error: {e}")
            raise e

    async def validate_utc_datetime_format(self, time_str: str, ctx: CSContext) -> None:
        """
        Validates that the given string is a strict UTC datetime: YYYY-MM-DDTHH:MM:SS.

        Args:
            time_str : UTC datetime string to validate
            ctx      : CSContext — for logging

        Raises:
            ValueError: If the format does not match.
        """
        try:
            import re as _re
            utc_pattern = _re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$")
            if not utc_pattern.match(time_str):
                raise ValueError(
                    f"Invalid 'time_in_utc' format: '{time_str}'. "
                    "Expected strict UTC format: YYYY-MM-DDTHH:MM:SS (e.g. '2026-08-01T09:00:00')"
                )
        except Exception as e:
            ctx.log.error(f"[Validation] validate_utc_datetime_format error: {e}")
            raise e

    async def validate_date_format(self, date_str: str, ctx: CSContext) -> None:
        """
        Validates that the given string is a real YYYY-MM-DD date.

        Checks:
          - Matches the YYYY-MM-DD regex pattern exactly
          - Is a real calendar date (e.g. 2026-02-30 will be rejected)

        Args:
            date_str : Date string to validate
            ctx      : CSContext — for logging

        Raises:
            ValueError: With a descriptive message if validation fails.
        """
        try:
            is_format_valid = self.DATE_PATTERN.match(date_str)
            if not is_format_valid:
                ctx.log.error(
                    f"[Validation] Invalid date format: '{date_str}'. Expected YYYY-MM-DD (example: 2026-04-20)"
                )
                return
            self.datetime.strptime(date_str, "%Y-%m-%d")
        except Exception as e:
            ctx.log.error(f"[Validation] validate_date_format error / not a real calendar date: {e}")
            raise e

    async def date_to_iso_string(self, date_str: str, ctx: CSContext) -> str:
        """
        Converts a YYYY-MM-DD string to a full ISO 8601 UTC datetime string.

        Args:
            date_str : Validated date string, e.g. "2026-04-20"
            ctx      : CSContext — for logging

        Returns:
            ISO string, e.g. "2026-04-20T00:00:00.000Z"
        """
        try:
            dt = self.datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=self.timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        except Exception as e:
            ctx.log.error(f"[Validation] date_to_iso_string error: {e}")
            raise e

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN RUN METHOD
    # ═══════════════════════════════════════════════════════════════════════

    async def run(self, config: dict, ctx: CSContext) -> dict:
        """
        Main entry point — routes logic based on the incoming category.

        Args:
            config (dict)  : Incoming automation request payload containing:
                - category      (str)      : Request category (drives which logic runs)
                - details       (str|dict) : JSON string or dict containing category-specific fields
            ctx (CSContext) : Context object used for logging

        Returns:
            dict: Result with status and relevant fields.

        Raises:
            Exception: Re-raises on any failure.

        Workflow:
            Reads category from config, then branches per category.
            On any error → raises the exception.
        """
        try:
            # ── Read category ─────────────────────────────────────────────
            category = config.get("category", "").strip()
            ctx.log.info(f"[run] Category received : '{category}'")

            # ── Parse details ─────────────────────────────────────────────
            details_str = config.get("details", "")
            try:
                import json
                if details_str:
                    # Remote \r and \n to handle copy-pasted or multiline Slack text
                    details_str_clean = details_str.replace('\r', '').replace('\n', '')
                    details = json.loads(details_str_clean)
                else:
                    details = {}
            except Exception as e:
                ctx.log.error(f"[run] Failed to parse details JSON: {e}. Raw str: {details_str!r}")
                details = {}

            # ── STEP 0: Initialize ArangoDB connection ───────────────────────
            self.arango_client = await self.connect_arango(ctx)
            # Default to ADMIN_DB for most collections like 'organization'
            self.db_client     = await self.connect_db(self.ADMIN_DB, ctx)

            # ── CATEGORY: Extend Trail Period ──────────────────────────────
            if category == "Extend Trail Period":

                # STEP 1: Validate required input fields
                org_id        = details.get("org_id", "").strip()
                extend_period = details.get("extend_period", "").strip()

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not extend_period:
                    raise ValueError("Missing required field in details: 'extend_period'")

                ctx.log.info(f"[ExtenTrailPeriod] Starting task for org_id='{org_id}'")
                ctx.log.info(f"[ExtenTrailPeriod] Requested expiry date  : {extend_period}")

                # STEP 2: Validate date format (YYYY-MM-DD only)
                await self.validate_date_format(extend_period, ctx)
                ctx.log.info(f"[ExtenTrailPeriod] Date format is valid    : {extend_period}")

                # STEP 3: Fetch org document — confirm org exists
                ctx.log.info(f"[ExtenTrailPeriod] Fetching org document for '{org_id}'...")
                org_doc = await self.get_org_document(org_id, ctx)

                if org_doc is None:
                    raise ValueError(
                        f"Organization not found in ArangoDB for org_id='{org_id}'"
                    )
                ctx.log.info(
                    f"[ExtenTrailPeriod] Org found : name='{org_doc.get('name', 'N/A')}'"
                )

                # STEP 4: Convert date → full ISO datetime string
                new_expiry_iso = await self.date_to_iso_string(extend_period, ctx)
                ctx.log.info(f"[ExtenTrailPeriod] ISO expiry datetime : {new_expiry_iso}")

                # STEP 5: Update ArangoDB — tierSettings.trialSettings.expiryTime
                ctx.log.info("[ExtenTrailPeriod] Updating ArangoDB...")
                success = await self.update_trial_expiry_time(org_id, new_expiry_iso, ctx)

                if not success:
                    raise Exception(
                        f"ArangoDB update returned no result for org_id='{org_id}'"
                    )

                result = {
                    "status":     "success",
                    "org_id":     org_id,
                    "new_expiry": new_expiry_iso,
                }

                ctx.log.info(f"[ExtenTrailPeriod] Task completed: {result}")
                return result

            elif category == "Update Refresh Time":
                # STEP 1: Validate required input fields
                org_id       = details.get("org_id", "").strip()
                timezone_str = details.get("timezone", "").strip()
                refresh_time = details.get("refreshTime")

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not timezone_str:
                    raise ValueError("Missing required field in details: 'timezone'")
                if not refresh_time or not isinstance(refresh_time, list):
                    raise ValueError("Missing or invalid required field in details: 'refreshTime' (must be a list of strings)")

                ctx.log.info(f"[UpdateRefreshTime] Starting task for org_id='{org_id}'")
                ctx.log.info(f"[UpdateRefreshTime] Requested timezone: {timezone_str}")
                ctx.log.info(f"[UpdateRefreshTime] Original requested refreshTime: {refresh_time}")

                # STEP 2: Validate timezone
                await self.validate_timezone(timezone_str, ctx)
                ctx.log.info(f"[UpdateRefreshTime] Timezone '{timezone_str}' is valid.")

                # STEP 3: Convert the array of local time strings to UTC ISO strings
                utc_refresh_times = []
                for rt in refresh_time:
                    converted = await self.convert_refresh_time_to_utc(rt, timezone_str, ctx)
                    utc_refresh_times.append(converted)
                ctx.log.info(f"[UpdateRefreshTime] Converted UTC refreshTime: {utc_refresh_times}")

                # STEP 4: Fetch org document — confirm org exists
                ctx.log.info(f"[UpdateRefreshTime] Fetching org document for '{org_id}'...")
                org_doc = await self.get_org_document(org_id, ctx)

                if org_doc is None:
                    raise ValueError(
                        f"Organization not found in ArangoDB for org_id='{org_id}'"
                    )
                ctx.log.info(f"[UpdateRefreshTime] Org found : name='{org_doc.get('name', 'N/A')}'")

                # STEP 5: Update ArangoDB — localeInfo.timezone and tierSettings.dataRefresh.refreshTime
                ctx.log.info("[UpdateRefreshTime] Updating ArangoDB...")
                success = await self.update_refresh_time(org_id, timezone_str, utc_refresh_times, ctx)

                if not success:
                    raise Exception(
                        f"ArangoDB update returned no result for org_id='{org_id}'"
                    )

                result = {
                    "status":      "success",
                    "org_id":      org_id,
                    "timezone":    timezone_str,
                    "refreshTime": utc_refresh_times,
                }

                ctx.log.info(f"[UpdateRefreshTime] Task completed: {result}")
                return result

            elif category == "Admin Email changes":
                # STEP 1: Validate required input fields
                role      = details.get("role", "").strip()
                old_email = details.get("old_email", "").strip().replace("mailto:", "")
                new_email = details.get("new_email", "").strip().replace("mailto:", "")

                # Admin required fields
                user_id    = details.get("user_id", "").strip() if details.get("user_id") else ""
                dataset_id = details.get("dataset_id", "").strip() if details.get("dataset_id") else ""
                org_id     = details.get("org_id", "").strip() if details.get("org_id") else ""

                if not role:
                    raise ValueError("Missing required field in details: 'role'")
                if not old_email:
                    raise ValueError("Missing required field in details: 'old_email'")
                if not new_email:
                    raise ValueError("Missing required field in details: 'new_email'")

                ctx.log.info(f"[AdminEmailChanges] Starting task for old_email='{old_email}'")

                # STEP 2: Execute email change logic
                success = await self.change_user_email_impl(
                    role=role,
                    old_email=old_email,
                    new_email=new_email,
                    user_id=user_id,
                    dataset_id=dataset_id,
                    org_id=org_id,
                    ctx=ctx
                )

                if not success:
                    raise Exception(f"Failed to change email for {old_email}")

                result = {
                    "status":    "success",
                    "role":      role,
                    "old_email": old_email,
                    "new_email": new_email,
                }

                ctx.log.info(f"[AdminEmailChanges] Task completed: {result}")
                return result

            elif category == "Enable Athena Threads":
                # STEP 1: Validate required input fields
                if isinstance(details, list):
                    org_list = details
                elif isinstance(details, dict) and "org_details" in details:
                    org_list = details.get("org_details", [])
                elif isinstance(details, dict) and "org_id" in details:
                    org_list = [details]
                else:
                    org_list = []

                if not org_list:
                    raise ValueError("Missing required fields in details: must provide org_id and dataset_id, or a list of such objects")

                template_org_id = 'decnavda'
                overall_success = True
                success_orgs = []
                failed_orgs = []

                for item in org_list:
                    org_id     = item.get("org_id", "").strip()
                    dataset_id = item.get("dataset_id", "").strip()
                    org_disp   = f"{org_id} (DS: {dataset_id})" if org_id and dataset_id else (org_id or "UNKNOWN")

                    if not org_id or not dataset_id:
                        ctx.log.error(f"[EnableAthenaThreads] Missing org_id or dataset_id in item: {item}")
                        overall_success = False
                        failed_orgs.append(org_disp)
                        continue

                    ctx.log.info(f"[EnableAthenaThreads] Starting task for org_id='{org_id}', dataset_id='{dataset_id}'")

                    # STEP 2: Execute enable athena threads logic
                    try:
                        success = await self.enable_athena_threads_impl(
                            org_id=org_id,
                            dataset_id=dataset_id,
                            template_org_id=template_org_id,
                            ctx=ctx
                        )
                        if success:
                            success_orgs.append(org_disp)
                        else:
                            overall_success = False
                            failed_orgs.append(org_disp)
                    except Exception as e:
                        ctx.log.error(f"[EnableAthenaThreads] Failed for {org_disp}: {e}")
                        overall_success = False
                        failed_orgs.append(org_disp)

                if not overall_success and not success_orgs:
                    raise Exception(f"Failed to enable athena threads for any orgs. Failed orgs: {failed_orgs}")

                result = {
                    "status":          "success" if overall_success else "partial_success",
                    "success_orgs":    success_orgs,
                    "failed_orgs":     failed_orgs,
                    "template_org_id": template_org_id,
                }

                ctx.log.info(f"[EnableAthenaThreads] Task completed: {result}")
                return result

            elif category == "Get Entity Count":
                # STEP 1: Validate required input fields
                tenant_id = details.get("tenant_id", "").strip()

                if not tenant_id:
                    raise ValueError("Missing required field in details: 'tenant_id'")

                ctx.log.info(f"[GetKBEntityCount] Starting task for tenant_id='{tenant_id}'")

                # STEP 2: Execute entity count logic
                count = await self.get_knowledge_base_entity_count(
                    tenant_id=tenant_id,
                    ctx=ctx
                )

                if count == -1:
                    raise Exception(f"Failed to retrieve knowledge base entity count for tenant '{tenant_id}'. Check logs for 403 or other HTTP errors.")

                result = {
                    "status":       "success",
                    "tenant_id":    tenant_id,
                    "entity_count": count,
                }

                ctx.log.info(f"[GetKBEntityCount] Task completed: {result}")
                return result

            # ── CATEGORY: Activate Dataset ────────────────────────────────
            elif category == "Activate Dataset":

                dataset_id = details.get("dataset_id", "").strip()
                org_id     = details.get("org_id", "").strip()
                schema_ds  = details.get("schema", {})

                if not dataset_id:
                    raise ValueError("Missing required field in details: 'dataset_id'")
                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not isinstance(schema_ds, dict) or not schema_ds.get("schema_to_activate", "").strip():
                    raise ValueError(
                        "Missing or invalid required field in details: 'schema' (must include 'schema_to_activate')"
                    )

                activate_current     = schema_ds.get("activate_current_schema", False)
                activate_in_progress = schema_ds.get("activate_in_progress_schema", False)
                activate_backup      = schema_ds.get("activate_backup_schema", False)

                flags_set = sum([bool(activate_current), bool(activate_in_progress), bool(activate_backup)])
                if flags_set != 1:
                    raise ValueError(
                        "Exactly one of 'activate_current_schema', 'activate_in_progress_schema', "
                        "or 'activate_backup_schema' must be true in details.schema"
                    )

                ctx.log.info(f"[ActivateDataset] Starting task for dataset_id='{dataset_id}', org_id='{org_id}'")

                result = {
                    "status":             "success",
                    "dataset_id":         dataset_id,
                    "org_id":             org_id,
                    "schema_to_activate": schema_ds.get("schema_to_activate"),
                }

                ctx.log.info(f"[ActivateDataset] Task completed: {result}")
                return result

            # ── CATEGORY: Remove SME Duplicates ───────────────────────────
            elif category == "Remove SME Duplicates":

                dataset_id               = details.get("dataset_id", "").strip()
                remove_synonym_duplicate  = bool(details.get("remove_synonym_duplicate", False))
                remove_metadata_duplicate = bool(details.get("remove_metadata_duplicate", False))

                if not dataset_id:
                    raise ValueError("Missing required field in details: 'dataset_id'")
                if not remove_synonym_duplicate and not remove_metadata_duplicate:
                    raise ValueError(
                        "At least one of 'remove_synonym_duplicate' or 'remove_metadata_duplicate' must be true"
                    )

                ctx.log.info(f"[RemoveSMEDuplicates] Starting task for dataset_id='{dataset_id}'")

                result = {
                    "status":                    "success",
                    "dataset_id":                dataset_id,
                    "remove_synonym_duplicate":  remove_synonym_duplicate,
                    "remove_metadata_duplicate": remove_metadata_duplicate,
                }

                ctx.log.info(f"[RemoveSMEDuplicates] Task completed: {result}")
                return result

            # ── CATEGORY: Increase Session Timeout ────────────────────────
            elif category == "Increase Session Timeout":

                org_id          = details.get("org_id", "").strip()
                time_in_minutes = str(details.get("time_in_minutes", "")).strip()

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not time_in_minutes:
                    raise ValueError("Missing required field in details: 'time_in_minutes'")

                ctx.log.info(f"[IncreaseSessionTimeout] Starting task for org_id='{org_id}', time_in_minutes='{time_in_minutes}'")

                result = {
                    "status":          "success",
                    "org_id":          org_id,
                    "time_in_minutes": time_in_minutes,
                }

                ctx.log.info(f"[IncreaseSessionTimeout] Task completed: {result}")
                return result

            # ── CATEGORY: Increase User Count ─────────────────────────────
            elif category == "Increase User Count":

                org_id     = details.get("org_id", "").strip()
                user_count = details.get("user_count", 0)

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not user_count or int(user_count) <= 0:
                    raise ValueError(
                        "Missing or invalid required field in details: 'user_count' (must be > 0)"
                    )

                user_count = int(user_count)

                ctx.log.info(f"[IncreaseUserCount] Starting task for org_id='{org_id}', user_count={user_count}")

                result = {
                    "status":     "success",
                    "org_id":     org_id,
                    "user_count": user_count,
                }

                ctx.log.info(f"[IncreaseUserCount] Task completed: {result}")
                return result

            # ── CATEGORY: Change Data Fetch Limit ─────────────────────────
            elif category == "Change Data Fetch Limit":

                dataset_id  = details.get("dataset_id", "").strip()
                fetch_limit = details.get("fetch_limit", 0)

                if not dataset_id:
                    raise ValueError("Missing required field in details: 'dataset_id'")
                if not fetch_limit or int(fetch_limit) <= 0:
                    raise ValueError(
                        "Missing or invalid required field in details: 'fetch_limit' (must be > 0)"
                    )

                fetch_limit = int(fetch_limit)

                ctx.log.info(f"[ChangeDataFetchLimit] Starting task for dataset_id='{dataset_id}', fetch_limit={fetch_limit}")

                result = {
                    "status":      "success",
                    "dataset_id":  dataset_id,
                    "fetch_limit": fetch_limit,
                }

                ctx.log.info(f"[ChangeDataFetchLimit] Task completed: {result}")
                return result

            # ── CATEGORY: Remove Insight Duplicates ───────────────────────
            elif category == "Remove Insight Duplicates":

                dataset_id = details.get("dataset_id", "").strip()

                if not dataset_id:
                    raise ValueError("Missing required field in details: 'dataset_id'")

                ctx.log.info(f"[RemoveInsightDuplicates] Starting task for dataset_id='{dataset_id}'")

                result = {
                    "status":     "success",
                    "dataset_id": dataset_id,
                }

                ctx.log.info(f"[RemoveInsightDuplicates] Task completed: {result}")
                return result

            # ── CATEGORY: Change Data Refresh Time ────────────────────────
            elif category == "Change Data Refresh Time":

                org_id      = details.get("org_id", "").strip()
                time_in_utc = details.get("time_in_utc", "").strip()

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not time_in_utc:
                    raise ValueError("Missing required field in details: 'time_in_utc'")

                await self.validate_utc_datetime_format(time_in_utc, ctx)

                ctx.log.info(f"[ChangeDataRefreshTime] Starting task for org_id='{org_id}', time_in_utc='{time_in_utc}'")

                result = {
                    "status":      "success",
                    "org_id":      org_id,
                    "time_in_utc": time_in_utc,
                }

                ctx.log.info(f"[ChangeDataRefreshTime] Task completed: {result}")
                return result

            # ── CATEGORY: Enable Connector V2 Menu ────────────────────────
            elif category == "Enable Connector V2 Menu":

                org_id     = details.get("org_id", "").strip()
                user_id    = details.get("user_id", "").strip()
                dataset_id = details.get("dataset_id", "").strip()

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not user_id:
                    raise ValueError("Missing required field in details: 'user_id'")
                if not dataset_id:
                    raise ValueError("Missing required field in details: 'dataset_id'")

                ctx.log.info(f"[EnableConnectorV2Menu] Starting task for org_id='{org_id}', user_id='{user_id}', dataset_id='{dataset_id}'")

                result = {
                    "status":     "success",
                    "org_id":     org_id,
                    "user_id":    user_id,
                    "dataset_id": dataset_id,
                }

                ctx.log.info(f"[EnableConnectorV2Menu] Task completed: {result}")
                return result

            # ── CATEGORY: Enable Athena Iq Menu ───────────────────────────
            elif category == "Enable Athena Iq Menu":

                org_id     = details.get("org_id", "").strip()
                user_id    = details.get("user_id", "").strip()
                dataset_id = details.get("dataset_id", "").strip()

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not user_id:
                    raise ValueError("Missing required field in details: 'user_id'")
                if not dataset_id:
                    raise ValueError("Missing required field in details: 'dataset_id'")

                ctx.log.info(f"[EnableAthenaIqMenu] Starting task for org_id='{org_id}', user_id='{user_id}', dataset_id='{dataset_id}'")

                result = {
                    "status":     "success",
                    "org_id":     org_id,
                    "user_id":    user_id,
                    "dataset_id": dataset_id,
                }

                ctx.log.info(f"[EnableAthenaIqMenu] Task completed: {result}")
                return result

            # elif category == "Create Organization":
            #     return await self.handle_create_organization(config, ctx)

            # elif category == "Reset Password":
            #     return await self.handle_reset_password(config, ctx)

            # ── UNKNOWN CATEGORY ───────────────────────────────────────────
            else:
                raise ValueError(
                    f"Unknown category: '{category}'. "
                    "No handler implemented for this category."
                )

        except Exception as e:
            ctx.log.error(f"[run] Error: {str(e)}")
            raise e

