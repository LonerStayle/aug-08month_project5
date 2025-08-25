from pathlib import Path
save_root = Path("./misclassified")  # 오분류 저장 루트
save_root.mkdir(parents=True, exist_ok=True)

def resolve_path(ds, idx):
    """
    Subset 계층을 거슬러 올라가 원본 dataset과 원본 idx로 바꾼 뒤
    (path, label) 정보를 뽑아 원본 파일 경로를 반환.
    """
    import torch.utils.data as tud
    while isinstance(ds, tud.Subset):
        idx = ds.indices[idx]
        ds = ds.dataset

    # torchvision.datasets.ImageFolder 계열
    if hasattr(ds, "samples") and len(ds.samples) > 0:
        path, _ = ds.samples[idx]
        return path
    if hasattr(ds, "imgs") and len(ds.imgs) > 0:
        path, _ = ds.imgs[idx]
        return path
    # 커스텀 데이터셋이라면 여기에 맞춰 수정
    if hasattr(ds, "files"):  # 예: ds.files[idx]
        return ds.files[idx]

    return None  # 경로를 모르면 None