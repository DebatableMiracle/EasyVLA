from huggingface_hub import HfApi, upload_file
import os

HF_REPO   = "honestlyanubhav/vla-from-scratch-day3"   # change this
HF_TOKEN  = os.environ.get("HF_TOKEN")      # set in env or paste directly
CKPT_PATH = "checkpoints/best.pt"

def push():
    api = HfApi()

    # create repo if it doesn't exist
    api.create_repo(repo_id=HF_REPO, token=HF_TOKEN, exist_ok=True)

    # push checkpoint
    api.upload_file(
        path_or_fileobj=CKPT_PATH,
        path_in_repo="checkpoints/best.pt",
        repo_id=HF_REPO,
        token=HF_TOKEN,
    )

    # push all python files
    for root, _, files in os.walk("."):
        for f in files:
            if f.endswith(".py") and ".git" not in root and "data" not in root:
                local = os.path.join(root, f)
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=local,
                    repo_id=HF_REPO,
                    token=HF_TOKEN,
                )
                print(f"Uploaded {local}")

    print(f"Pushed to https://huggingface.co/{HF_REPO}")

if __name__ == "__main__":
    push()