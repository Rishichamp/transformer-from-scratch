"""Entry point. Usage: python run.py"""
import uvicorn
from app.config import settings
if __name__ == "__main__":
    print(f"\n🔬  Transformer Architecture Lab")
    print(f"   Positional encoding : {settings.POS_ENCODING}")
    print(f"   Activation          : {settings.ACTIVATION}")
    print(f"   d_model / heads     : {settings.D_MODEL} / {settings.N_HEADS}")
    print(f"   Docs                : http://127.0.0.1:{settings.APP_PORT}/docs")
    print(f"\n   If no checkpoint exists, serve an untrained model for architecture inspection.")
    print(f"   Train: python scripts/train.py\n")
    uvicorn.run("app.main:app", host=settings.APP_HOST, port=settings.APP_PORT, log_level="info")
