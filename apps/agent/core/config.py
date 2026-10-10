from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str
    # Product version — overridden at deploy time via APP_VERSION (falls back to "dev").
    app_version: str = "dev"
    # Sentry error monitoring — empty DSN disables it (local dev/CI stay offline).
    sentry_dsn: str = ""
    sentry_environment: str = "development"
    # PostHog product analytics — empty key disables it (local dev/CI stay offline).
    posthog_api_key: str = ""
    posthog_host: str = "https://eu.i.posthog.com"
    aws_region: str = "eu-central-1"
    # Custom EventBridge bus the app publishes onto. Defaults to the literal prod/local
    # bus name (unsuffixed) so the agent can publish without a wired env var — the agent
    # task doesn't set EVENTBRIDGE_BUS_NAME (unlike api/worker), and this is the only bus.
    eventbridge_bus_name: str = "jev-bakeoff"
    agent_queue_url: str
    # How many SQS messages the agent processes concurrently. The consumer is a single
    # task, so without this a slow handler (rate scrape, Playwright fetch, offer/PDF
    # extraction) blocks every fast one (status line, validation) behind it — head-of-line
    # starvation that makes a 2-4s status call feel like it "hangs". Bounded so we never
    # open more DB sessions than the pool (default 5 ≤ pool_size 5 + overflow 10) or hit
    # Anthropic rate limits. Raise once heavy jobs get split into their own lane/queue.
    agent_concurrency: int = 5
    s3_bucket: str
    anthropic_api_key: str
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_endpoint_url: str | None = None
    s3_endpoint_url: str | None = None
    api_internal_url: str = "http://localhost:8000"
    assisted_fetch_enabled: bool = False
    # Gate for the property-analysis agent's live listing fetch. Default off so the
    # xe.gr / spitogatos.gr fetch stays disabled in prod until legal signs off on the
    # sites' ToS + GDPR wording for storing listing photos (mirrors assisted_fetch).
    listing_analysis_enabled: bool = False
    # Gate for the rate-scraper agent's live fetch of banks' published rates pages.
    # Enabled — ToS sign-off given for automated daily reads. Nothing is actually fetched
    # until an admin fills a verified rate_page_url per bank (all ship NULL).
    bank_rate_scrape_enabled: bool = True
    # Gate for the firm-branding agent's live fetch of a firm's own website (to
    # design a color theme). Default on: unlike listing/rate fetches it reads the
    # broker's OWN public homepage (they typed the URL), the same low-risk public
    # preview stance as property_analysis. Flip off as a kill switch if needed.
    firm_branding_enabled: bool = True
    internal_token: str = ""
    # Gate for best-effort real-time push to the API's WebSocket registry
    # (POST /internal/notify). Default off so local dev / CI make no such call;
    # realtime is purely advisory and never affects validation when disabled.
    realtime_enabled: bool = False
    # Gate for the market-radar agent's live fetch of Greek news/gov sources
    # (agents/market_radar_sources.py). Default OFF: nothing is fetched until we
    # confirm each source's ToS posture (official RSS + link-only for © media/banks).
    market_radar_enabled: bool = False
    # Cheap, high-volume triage model for the market-radar agent (it scores/summarises
    # many candidate headlines per tick). Haiku keeps the per-tick cost low.
    market_radar_model: str = "claude-haiku-4-5-20251001"
    # Safety caps for one tick: how many fresh candidates to summarise (cost bound) and
    # the minimum relevance the LLM must assign for an item to be stored.
    market_radar_max_items: int = 40
    market_radar_min_relevance: float = 0.35
    # Gate for the news → client relevancy matcher (docs/MARKET_RADAR_MATCHER.md): when a
    # stored news item carries a structured signal, publish ``radar.signal.detected`` and run
    # the matcher that flags affected clients. Default OFF so triage can extract + store signals
    # (harmless, additive) while the publish+match path stays inert until the UI phase ships.
    market_radar_signals_enabled: bool = False
    # Law 128/75 levy (housing loans) added on top of every effective rate the
    # offer-recommendation producer prices, mirroring the API calculator
    # (apps/api/core/config.py mortgage_law128_levy_pct). Percentage (0.12 == 0.12%).
    mortgage_law128_levy_pct: float = 0.12
    # Default loan-to-value the offer-recommendation producer assumes when sizing the
    # loan from the property price. 80% is the standard Greek primary-residence cap.
    offer_rec_default_ltv_pct: float = 80.0
    # Default term (years) the offer-recommendation producer amortizes over when
    # ranking banks. 25y is the common Greek mortgage term.
    offer_rec_default_term_years: int = 25
    # Gate for the Case Navigator (agents/case_navigator.py): the per-case agent that
    # perceives a case, decides ONE Next Best Action, and PROPOSES it as a
    # next_best_action suggestion (no autonomous send). Default OFF — nothing runs until
    # a navigator.tick arrives AND this is on (docs/CASE_NAVIGATOR_SLICE1.md).
    case_navigator_enabled: bool = False
    # Cheap-by-default reasoning model for the per-case tick. The last_perceived_hash gate
    # already skips ticks that changed nothing, so most wakes never reach the model.
    case_navigator_model: str = "claude-haiku-4-5-20251001"
    # Days a nudged-but-silent client waits before the scheduled sweep re-nudges — the cadence
    # that lets a case advance (wait → re-nudge → escalate) even when the board hasn't changed.
    case_navigator_renudge_days: int = 3
    # Gate for the desk_manager aggregation agent (agents/desk_manager.py, Phase 2 P2.2a):
    # the deterministic "manager tier" that distils per-desk signals into the desk_context
    # read-model the per-case Navigator already reads (P2.1). Default OFF — nothing runs until
    # a desk_manager.sweep/tick arrives AND this is on, so desk_context stays empty and the
    # Navigator behaves exactly as today (docs/CASE_NAVIGATOR_PHASE2.md).
    desk_manager_enabled: bool = False
    # Minimum number of reviews a (desk, document_type) pair needs before its reject-rate is
    # trusted enough to publish into desk_context.tendencies — below this we omit the type
    # rather than emit a noisy rate off one or two reviews.
    desk_manager_min_reviews: int = 3
    # Window (days) after a navigator nudge within which an upload counts as a "response" for
    # the nudge→response-rate tendency (P2.2b). An upload later than this is not credited to the
    # nudge (the client may have moved for an unrelated reason).
    desk_manager_response_window_days: int = 7
    # Minimum number of nudges a (desk, channel) pair needs before its nudge→response rate is
    # published into desk_context.tendencies.nudge_response — below this we omit the channel
    # rather than emit a rate off one or two nudges (mirrors desk_manager_min_reviews' stance).
    desk_manager_min_nudges: int = 3


settings = Settings()  # type: ignore[call-arg]  # fields populated from env at runtime
