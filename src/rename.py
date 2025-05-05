import os

# Path to your dataset
dataset_path = r"D:\1.Ebooks\4th Year 1st Sem\Phase 1 Project\Skin Disease Detetction using CNN\dataset"

# Folder renaming mapping
folder_rename_map = {
    "acne disease": "diseased_acne",
    "Dermatitis dissese": "diseased_dermatitis",
    "eczema disease": "diseased_eczema",
    "Healthy skin": "healthy_skin",
    "melanoma": "diseased_melanoma",
    "psoriasis disease": "diseased_psoriasis"
}

# Rename the folders
for old_name, new_name in folder_rename_map.items():
    old_folder = os.path.join(dataset_path, old_name)
    new_folder = os.path.join(dataset_path, new_name)
    
    if os.path.exists(old_folder):
        os.rename(old_folder, new_folder)
        print(f"Renamed: {old_name} → {new_name}")
    else:
        print(f"Folder {old_name} does not exist.")

print("Folder renaming complete!")
