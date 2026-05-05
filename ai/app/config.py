from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8100
    debug: bool = True

    models_dir: str = "/models"
    default_local_model: str = ""  # empty = nothing pre-loaded
    local_llm_threads: int = 4
    n_ctx: int = 4096

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
