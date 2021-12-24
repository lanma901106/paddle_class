

def colorize(gray, palette):
    # gray: numpy array of the label and 1*3N size list palette
    color = Image.fromarray(gray.astype(np.uint8)).convert('P')
    color.putpalette(palette)
    return color


def save_blend_image(image_file, pred_file):
    image1 = Image.open(image_file)
    image2 = Image.open(pred_file)
    image1 = image1.convert('RGBA')
    image2 = image2.convert('RGBA')
    image = Image.blend(image1, image2, 0.5)
    o_file = pred_file[0:-4] + "_blend.png"
    image.save(o_file)




def inference_resize()

def inference_sliding()

def inference_multi_scale()



def save_images



# this inference code reads a list of image path, and do prediction for each image one by one
def main():
    # 0. env preparation

    # 1. create model

    # 2. load pretrained model

    # 3. read test image list

    # 4. create transforms for test image, transform should be same as training

    # 5. loop over list of images

        # 6. read image and do preprocessing

        # 7. image to variable

        # 8. call inference func

        # 9. save results

if __name__ == "__main__":
    main()
