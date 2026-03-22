import os

def count_images(folder_path):
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp')

    count = 0
    for file in os.listdir(folder_path):
        if file.lower().endswith(image_extensions):
            count += 1

    return count


folder = "output_fixed"
print("Số ảnh:", count_images(folder))