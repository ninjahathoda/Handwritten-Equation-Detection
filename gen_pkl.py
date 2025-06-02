import pickle as pkl
import numpy
import imageio.v2 as imageio

# Corrected path
image_path = 'C:\\Users\\harsh\\OneDrive\\Desktop\\AMER\\off_image_train\\off_image_train\\off_image_train\\'
outFile = 'C:\\Users\\harsh\\OneDrive\\Desktop\\AMER\\offline-train.pkl'
oupFp_feature = open(outFile, 'wb')

features = {}
channels = 1
sentNum = 0

scpFile = open('C:\\Users\\harsh\\OneDrive\\Desktop\\AMER\\train_caption.txt')
while True:
    line = scpFile.readline().strip()
    if not line:
        break
    else:
        key = line.split(None, 1)[0]  # Split on any whitespace
        # Append '_0.bmp' instead of '.0.bmp' to match filename
        image_file = image_path + key + '_0.bmp'
        try:
            im = imageio.imread(image_file)
            mat = numpy.zeros([channels, im.shape[0], im.shape[1]], dtype='uint8')
            mat[0, :, :] = im
            sentNum += 1
            features[key] = mat
            if sentNum % 500 == 0:
                print('Process sentences', sentNum)
            print(f"Processed: {key}_0.bmp")
        except FileNotFoundError:
            print(f"Warning: Image not found: {image_file}")
            continue
print('Load images done. Sentence number', sentNum)
pkl.dump(features, oupFp_feature)
print('Save file done')
oupFp_feature.close()
scpFile.close()