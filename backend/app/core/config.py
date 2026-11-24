from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "FinPilot AI"
    database_url: str = "sqlite:///./finpilot.db"
    secret_key: str = "change-me-in-production"
    access_token_expire_minutes: int = 60 * 24 * 7
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Set to true to seed a labelled demo account on startup
    seed_demo: bool = False

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


settings = Settings()
