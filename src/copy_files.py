import shutil
import os

file_names = [
    "uid_vid_00000.mp4", "uid_vid_00003.mp4", "uid_vid_00004.mp4", "uid_vid_00006.mp4",
    "uid_vid_00008.mp4", "uid_vid_00010.mp4", "uid_vid_00011.mp4", "uid_vid_00012.mp4",
    "uid_vid_00014.mp4", "uid_vid_00015.mp4", "uid_vid_00016.mp4", "uid_vid_00018.mp4",
    "uid_vid_00019.mp4", "uid_vid_00020.mp4", "uid_vid_00021.mp4", "uid_vid_00022.mp4",
    "uid_vid_00024.mp4", "uid_vid_00027.mp4", "uid_vid_00030.mp4", "uid_vid_00031.mp4",
    "uid_vid_00034.mp4", "uid_vid_00035.mp4", "uid_vid_00036.mp4", "uid_vid_00040.mp4",
    "uid_vid_00041.mp4", "uid_vid_00042.mp4", "uid_vid_00043.mp4", "uid_vid_00046.mp4",
    "uid_vid_00050.mp4", "uid_vid_00052.mp4", "uid_vid_00053.mp4", "uid_vid_00057.mp4",
    "uid_vid_00064.mp4", "uid_vid_00065.mp4", "uid_vid_00066.mp4", "uid_vid_00078.mp4",
    "uid_vid_00080.mp4", "uid_vid_00084.mp4", "uid_vid_00086.mp4", "uid_vid_00087.mp4",
    "uid_vid_00088.mp4", "uid_vid_00094.mp4", "uid_vid_00096.mp4", "uid_vid_00099.mp4",
    "uid_vid_00103.mp4", "uid_vid_00104.mp4", "uid_vid_00105.mp4", "uid_vid_00106.mp4",
    "uid_vid_00108.mp4", "uid_vid_00111.mp4", "uid_vid_00115.mp4", "uid_vid_00121.mp4",
    "uid_vid_00129.mp4", "uid_vid_00140.mp4", "uid_vid_00144.mp4", "uid_vid_00147.mp4",
    "uid_vid_00158.mp4", "uid_vid_00165.mp4", "uid_vid_00175.mp4", "uid_vid_00176.mp4",
    "uid_vid_00183.mp4", "uid_vid_00186.mp4", "uid_vid_00190.mp4", "uid_vid_00191.mp4",
    "uid_vid_00196.mp4", "uid_vid_00197.mp4", "uid_vid_00200.mp4", "uid_vid_00201.mp4",
    "uid_vid_00207.mp4", "uid_vid_00209.mp4", "uid_vid_00215.mp4", "uid_vid_00218.mp4",
    "uid_vid_00219.mp4", "uid_vid_00221.mp4", "uid_vid_00222.mp4", "uid_vid_00224.mp4",
    "uid_vid_00225.mp4", "uid_vid_00226.mp4", "uid_vid_00227.mp4"
]

def copy_selected_txt_files(source_dir, destination_dir, file_list):
    
    # Upewniamy się, że katalog docelowy istnieje
    os.makedirs(destination_dir, exist_ok=True)
    
    for file_name in file_list:
        file_name += ".json"
        source_path = os.path.join(source_dir, file_name)
        destination_path = os.path.join(destination_dir, file_name)
        
        # Sprawdzamy, czy plik istnieje w katalogu źródłowym
        if os.path.exists(source_path) and file_name.endswith(".mp4.json"):
            shutil.copy2(source_path, destination_path)
            print(f"Skopiowano: {file_name}")
        else:
            print(f"Pominięto (nie istnieje lub zły format): {file_name}")

# Przykładowe użycie
source_directory = "/Users/grzegorzsmereczniak/Downloads/tracking-dataset-main/dataset/personpath22/annotation/anno_visible_2022"
destination_directory = "annotation"

copy_selected_txt_files(source_directory, destination_directory, file_names)
