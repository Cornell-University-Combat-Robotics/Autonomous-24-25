import cv2
from vid_to_warped_frames import get_homography_mat


def warp_image(image_name, output_w=600, output_h=600):

    input_path = INPUT_BASE_PATH + image_name
    image = cv2.imread(input_path)

    homography_mat = get_homography_mat(image, output_w, output_h)

    warped_image = cv2.warpPerspective(
        image, homography_mat, (output_w, output_h))

    output_path = OUTPUT_BASE_PATH + "warped_" + image_name
    cv2.imwrite(output_path, warped_image)


if __name__ == "__main__":
    INPUT_BASE_PATH = "data/input_img/"
    OUTPUT_BASE_PATH = "data/warped_img/"
    img_name = "test_img.png"
    warp_image(img_name)
