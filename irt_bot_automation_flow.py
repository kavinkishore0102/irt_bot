@task
class IRTBotAutomation:
    """
    IRTAutomation task class validates inputs for each automation category
    without performing any Jira or Slack updates.

    This class manages:
    - Input validation for all supported automation categories
    - Returns a success message with validated fields when inputs are valid

    Attributes:
        config : Configuration settings (None by default)
    """

    import re
    import json
    from typing import Optional
    from datetime import datetime, timezone

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
        self.config = None

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATION HELPER METHODS
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

    async def validate_email(self, email: str, label: str, ctx: CSContext) -> None:
        """
        Basic email format validation.

        Args:
            email  : Email string to validate
            label  : Field label for error messages (e.g. 'old_email')
            ctx    : CSContext — for logging

        Raises:
            ValueError: If the email is empty or missing '@'.
        """
        try:
            if not email or "@" not in email:
                raise ValueError(f"Invalid email in field '{label}': '{email}'")
        except Exception as e:
            ctx.log.error(f"[Validation] validate_email error: {e}")
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

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN RUN METHOD
    # ═══════════════════════════════════════════════════════════════════════

    async def run(self, config: dict, ctx: CSContext) -> dict:
        """
        Main entry point — routes validation logic based on the incoming category.

        No Jira or Slack updates are performed. Only the required inputs for each
        category are validated. On success a dict with status='success' and a
        human-readable message is returned.

        Args:
            config (dict)  : Incoming automation request payload containing:
                - category      (str) : Request category (drives which validation runs)
                - details       (dict|list) : JSON object containing category-specific fields
            ctx (CSContext) : Context object used for logging

        Returns:
            dict: {"status": "success", "message": str, ...validated_fields}

        Raises:
            ValueError: On missing / invalid input fields.
        """
        try:
            # ── Read category ─────────────────────────────────────────────
            category = config.get("category", "").strip()
            ctx.log.info(f"[IRTAutomation] Category received: '{category}'")

            # ── Fetch details ─────────────────────────────────────────────
            details = config.get("details", {})
            if isinstance(details, str):
                try:
                    import json
                    if details.strip():
                        details = json.loads(details.replace('\r', '').replace('\n', ''))
                    else:
                        details = {}
                except Exception as e:
                    ctx.log.error(f"[IRTAutomation] Failed to parse details JSON string: {e}")
                    details = {}

            # ── CATEGORY: Extend Trail Period ──────────────────────────────
            if category == "Extend Trail Period":

                org_id        = details.get("org_id", "").strip()
                extend_period = details.get("extend_period", "").strip()

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not extend_period:
                    raise ValueError("Missing required field in details: 'extend_period'")

                await self.validate_date_format(extend_period, ctx)

                ctx.log.info(f"[IRTAutomation][ExtendTrailPeriod] Validation passed — org_id='{org_id}', extend_period='{extend_period}'")
                return {
                    "status":         "success",
                    "message":        f"Input validation successful for category '{category}'.",
                    "org_id":         org_id,
                    "extend_period":  extend_period,
                }

            # ── CATEGORY: Update Refresh Time ──────────────────────────────
            elif category == "Update Refresh Time":

                org_id       = details.get("org_id", "").strip()
                timezone_str = details.get("timezone", "").strip()
                refresh_time = details.get("refreshTime")

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not timezone_str:
                    raise ValueError("Missing required field in details: 'timezone'")
                if not refresh_time or not isinstance(refresh_time, list):
                    raise ValueError(
                        "Missing or invalid required field in details: 'refreshTime' (must be a list of strings)"
                    )

                await self.validate_timezone(timezone_str, ctx)

                ctx.log.info(f"[IRTAutomation][UpdateRefreshTime] Validation passed — org_id='{org_id}', timezone='{timezone_str}'")
                return {
                    "status":      "success",
                    "message":     f"Input validation successful for category '{category}'.",
                    "org_id":      org_id,
                    "timezone":    timezone_str,
                    "refreshTime": refresh_time,
                }

            # ── CATEGORY: Admin Email changes ──────────────────────────────
            elif category == "Admin Email changes":

                role      = details.get("role", "").strip()
                old_email = details.get("old_email", "").strip().replace("mailto:", "")
                new_email = details.get("new_email", "").strip().replace("mailto:", "")

                # Admin-only required fields
                user_id    = details.get("user_id", "").strip()    if details.get("user_id")    else ""
                dataset_id = details.get("dataset_id", "").strip() if details.get("dataset_id") else ""
                org_id     = details.get("org_id", "").strip()     if details.get("org_id")     else ""

                if not role:
                    raise ValueError("Missing required field in details: 'role'")
                if not old_email:
                    raise ValueError("Missing required field in details: 'old_email'")
                if not new_email:
                    raise ValueError("Missing required field in details: 'new_email'")

                await self.validate_email(old_email, "old_email", ctx)
                await self.validate_email(new_email, "new_email", ctx)

                if role.lower() == "admin":
                    if not user_id:
                        raise ValueError("Admin role requires 'user_id' in details")
                    if not dataset_id:
                        raise ValueError("Admin role requires 'dataset_id' in details")
                    if not org_id:
                        raise ValueError("Admin role requires 'org_id' in details")

                ctx.log.info(f"[IRTAutomation][AdminEmailChanges] Validation passed — role='{role}', old_email='{old_email}'")
                result = {
                    "status":    "success",
                    "message":   f"Input validation successful for category '{category}'.",
                    "role":      role,
                    "old_email": old_email,
                    "new_email": new_email,
                }
                if role.lower() == "admin":
                    result.update({"user_id": user_id, "dataset_id": dataset_id, "org_id": org_id})
                return result

            # ── CATEGORY: Enable Athena Threads ───────────────────────────
            elif category == "Enable Athena Threads":

                if isinstance(details, list):
                    org_list = details
                elif isinstance(details, dict) and "org_details" in details:
                    org_list = details.get("org_details", [])
                elif isinstance(details, dict) and "org_id" in details:
                    org_list = [details]
                else:
                    org_list = []

                if not org_list:
                    raise ValueError(
                        "Missing required fields in details: must provide org_id and dataset_id, "
                        "or a list of such objects"
                    )

                valid_items   = []
                invalid_items = []

                for item in org_list:
                    org_id     = item.get("org_id", "").strip()
                    dataset_id = item.get("dataset_id", "").strip()
                    if not org_id or not dataset_id:
                        invalid_items.append(item)
                    else:
                        valid_items.append({"org_id": org_id, "dataset_id": dataset_id})

                if not valid_items:
                    raise ValueError(
                        f"All items in org_list are missing org_id or dataset_id. Invalid items: {invalid_items}"
                    )

                ctx.log.info(f"[IRTAutomation][EnableAthenaThreads] Validation passed — {len(valid_items)} valid org(s)")
                return {
                    "status":        "success",
                    "message":       f"Input validation successful for category '{category}'.",
                    "valid_orgs":    valid_items,
                    "invalid_orgs":  invalid_items,
                }

            # ── CATEGORY: Get Entity Count ─────────────────────────────────
            elif category == "Get Entity Count":

                tenant_id = details.get("tenant_id", "").strip()

                if not tenant_id:
                    raise ValueError("Missing required field in details: 'tenant_id'")

                ctx.log.info(f"[IRTAutomation][GetEntityCount] Validation passed — tenant_id='{tenant_id}'")
                return {
                    "status":    "success",
                    "message":   f"Input validation successful for category '{category}'.",
                    "tenant_id": tenant_id,
                }

            # ── CATEGORY: Activate Dataset ─────────────────────────────────
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

                ctx.log.info(f"[IRTAutomation][ActivateDataset] Validation passed — dataset_id='{dataset_id}', org_id='{org_id}'")
                return {
                    "status":              "success",
                    "message":             f"Input validation successful for category '{category}'.",
                    "dataset_id":          dataset_id,
                    "org_id":              org_id,
                    "schema_to_activate":  schema_ds.get("schema_to_activate"),
                }

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

                ctx.log.info(f"[IRTAutomation][RemoveSMEDuplicates] Validation passed — dataset_id='{dataset_id}'")
                return {
                    "status":                    "success",
                    "message":                   f"Input validation successful for category '{category}'.",
                    "dataset_id":                dataset_id,
                    "remove_metadata_duplicate": remove_metadata_duplicate,
                    "remove_synonym_duplicate":  remove_synonym_duplicate,
                }

            # ── CATEGORY: Increase Session Timeout ────────────────────────
            elif category == "Increase Session Timeout":

                org_id          = details.get("org_id", "").strip()
                time_in_minutes = str(details.get("time_in_minutes", "")).strip()

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not time_in_minutes:
                    raise ValueError("Missing required field in details: 'time_in_minutes'")

                ctx.log.info(f"[IRTAutomation][IncreaseSessionTimeout] Validation passed — org_id='{org_id}', time_in_minutes='{time_in_minutes}'")
                return {
                    "status":          "success",
                    "message":         f"Input validation successful for category '{category}'.",
                    "org_id":          org_id,
                    "time_in_minutes": time_in_minutes,
                }

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

                ctx.log.info(f"[IRTAutomation][IncreaseUserCount] Validation passed — org_id='{org_id}', user_count={user_count}")
                return {
                    "status":     "success",
                    "message":    f"Input validation successful for category '{category}'.",
                    "org_id":     org_id,
                    "user_count": user_count,
                }

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

                ctx.log.info(f"[IRTAutomation][ChangeDataFetchLimit] Validation passed — dataset_id='{dataset_id}', fetch_limit={fetch_limit}")
                return {
                    "status":      "success",
                    "message":     f"Input validation successful for category '{category}'.",
                    "dataset_id":  dataset_id,
                    "fetch_limit": fetch_limit,
                }

            # ── CATEGORY: Remove Insight Duplicates ───────────────────────
            elif category == "Remove Insight Duplicates":

                dataset_id = details.get("dataset_id", "").strip()

                if not dataset_id:
                    raise ValueError("Missing required field in details: 'dataset_id'")

                ctx.log.info(f"[IRTAutomation][RemoveInsightDuplicates] Validation passed — dataset_id='{dataset_id}'")
                return {
                    "status":     "success",
                    "message":    f"Input validation successful for category '{category}'.",
                    "dataset_id": dataset_id,
                }

            # ── CATEGORY: Change Data Refresh Time ────────────────────────
            elif category == "Change Data Refresh Time":

                org_id      = details.get("org_id", "").strip()
                time_in_utc = details.get("time_in_utc", "").strip()

                if not org_id:
                    raise ValueError("Missing required field in details: 'org_id'")
                if not time_in_utc:
                    raise ValueError("Missing required field in details: 'time_in_utc'")

                await self.validate_utc_datetime_format(time_in_utc, ctx)

                ctx.log.info(f"[IRTAutomation][ChangeDataRefreshTime] Validation passed — org_id='{org_id}', time_in_utc='{time_in_utc}'")
                return {
                    "status":      "success",
                    "message":     f"Input validation successful for category '{category}'.",
                    "org_id":      org_id,
                    "time_in_utc": time_in_utc,
                }

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

                ctx.log.info(f"[IRTAutomation][EnableConnectorV2Menu] Validation passed — org_id='{org_id}', user_id='{user_id}', dataset_id='{dataset_id}'")
                return {
                    "status":     "success",
                    "message":    f"Input validation successful for category '{category}'.",
                    "org_id":     org_id,
                    "user_id":    user_id,
                    "dataset_id": dataset_id,
                }

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

                ctx.log.info(f"[IRTAutomation][EnableAthenaIqMenu] Validation passed — org_id='{org_id}', user_id='{user_id}', dataset_id='{dataset_id}'")
                return {
                    "status":     "success",
                    "message":    f"Input validation successful for category '{category}'.",
                    "org_id":     org_id,
                    "user_id":    user_id,
                    "dataset_id": dataset_id,
                }

            # ── UNKNOWN CATEGORY ───────────────────────────────────────────
            else:
                raise ValueError(
                    f"Unknown category: '{category}'. "
                    "No handler implemented for this category."
                )

        except Exception as e:
            error_msg = str(e)
            ctx.log.error(f"[IRTAutomation] Validation error: {error_msg}")
            raise e
