from PIL import Image, ImageDraw

# Create a 256x256 image with a dark grey background
img = Image.new('RGB', (256, 256), color=(50, 50, 50))
draw = ImageDraw.Draw(img)

# Draw a bright green rectangle (representing an encroachment)
# Coordinates: x1, y1, x2, y2
draw.rectangle([100, 100, 150, 150], fill=(0, 255, 0), outline=(255, 255, 255))

# Save the file
img.save('test_sample.jpg')
print("Successfully created test_sample.jpg!")