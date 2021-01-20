from PIL import Image;

filename = input();
f = open(filename, 'r');
a = list(map(int, f.read().split()));

im = Image.new('RGB', (1920, 1080));
px = im.load();

top = 0;
for i in range(1920):
    top = max(top, a[2 * i + 1]);
print(top);

for i in range(1920):
    for j in range(1000):
        if j / 1000 * top <= a[2 * i + 1]:
            px[i, 1040 - j] = 0, 0, 255;
        else:
            px[i, 1040 - j] = 255, 255, 255;
    if i % 256 == 0:
        for j in range(20):
            px[i, 1050 + j] = 255, 0, 0;
            px[i, 10 + j] = 255, 0, 0;

im.save('image.bmp');