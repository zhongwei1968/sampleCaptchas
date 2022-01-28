import os
import numpy as np
# load an image with Pillow
from PIL import Image

class Captcha(object):

    # static variable to store character morphology map
    chars_morph_map = None

    def __init__(self):
        # load character morphologies only one time
        if Captcha.chars_morph_map is None:
            # get current script directory
            script = os.path.realpath(__file__)
            dirname = os.path.dirname(script)
            # the input and output of sampleCaptchas are stored at current script directory
            Captcha.chars_morph_map = Captcha.load_chars_morph_data(dirname)

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        morphs_read = Captcha.read_morphs_from_image(im_path)

        ch_list = []
        for morph in morphs_read:
            ch = Captcha.infer_char_from_morph(morph)
            if ch:
                ch_list.append(ch)

        with open(save_path, 'w') as f:
            f.write(''.join(ch_list))
        #print(''.join(ch_list))


    def infer_char_from_morph(morph):
        """
        Algo for inference a character from the morphology
        args:
            morph: morphology array, size[10,8], with True, False values
        return: the inference character
        """
        for ch in Captcha.chars_morph_map:
            # check if the morph is same as any morph for the character
            if any((morph_x == morph).all() for morph_x in  Captcha.chars_morph_map[ch]):
                return ch
        return None

    def read_morphs_from_image(img_file):
        """
        algo to retrieve the character morphology list from a image file
        args:
            img_file: .jpg image file to load
        return: morphology list
        """

        char_morphs = []
        try:
            image = Image.open(img_file)
            #convert image data to array
            data_rgb = np.asarray(image)

            #slice image array only with R value, the values for RGB are same for these image data.
            data_r = data_rgb[:, :, 0]

            # remove the frame area
            data_display = data_r[11:21, 5:50]

            # As background texture color values are all greater than 100
            # remove the background texture,  set as False (white)
            # set the foreground  as True (black)
            morph_area__bw = data_display < 100

            # retrieve 5 character morphologies, the height is 10, the width is 8, space width is 1
            for j in range(5):
                morph_bw = morph_area__bw[:, j * 9:j * 9 + 8]
                char_morphs.append(morph_bw)
        except Exception as e:
            print(e)
        return char_morphs

    def load_chars_morph_data(data_path):
        chars_morph_map = {}

        # read the image training samples one by one
        for i in range(25):
            try:
                # read the text file and get the characters text
                txt_file = data_path + '/output/output' + str(i).zfill(2) + '.txt'
                with open(txt_file) as f:
                    text_chars = f.readline().strip()

                # read the image and morphologies
                img_file = data_path + '/input/input' + str(i).zfill(2) + '.jpg'
                img_morphs = Captcha.read_morphs_from_image(img_file)

                for ch, morph in zip(text_chars, img_morphs):
                    if ch in chars_morph_map:
                        # check if the morph is same as any morph for the character
                        if not any( (morph_x==morph).all() for morph_x in chars_morph_map[ch]):
                            # we would support multiple morphs for same character
                            chars_img_map[ch].append(morph)
                    else:
                        chars_morph_map[ch] = [morph]

            except Exception as e:
                pass
                #print(e)

        return chars_morph_map


if __name__ == "__main__":
    my_captcha = Captcha()
    my_captcha('input/input100.jpg', 'output/output100.txt')
