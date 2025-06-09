from PIL import Image, ImageDraw, ImageFont


def expand_list_to_match_lods(lst, lods):
    length = len(lst)
    repeats = lods - length
    return lst + [lst[-1]] * repeats


def combine_images_with_titles_and_save(images, titles, output_path):

    width, height = images[0].shape[1], images[0].shape[0]
    total_height = height + 50  
    combined_image = Image.new('RGB', (width * len(images), total_height), color='white')

    draw = ImageDraw.Draw(combined_image)
    font = ImageFont.truetype("/opt/conda/pkgs/pillow-9.2.0-py37h850a105_2/info/test/Tests/fonts/FreeMono.ttf", size=40)
    
    for i, (img, title) in enumerate(zip(images, titles)):
        combined_image.paste(Image.fromarray(img), (i * width, 0))
        title_position = ((i * width) + width // 2 - draw.textlength(title, font=font) // 2, height)
        draw.text(title_position, title, font=font, fill='black')

    combined_image.save(output_path)
