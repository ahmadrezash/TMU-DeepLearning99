import cv2
import numpy as np


def compute_kernel(img='Image.jpg', kernel=None, out_name="result4"):
	if kernel is None:
		kernel = [[0, -1, 0],
				  [-1, 5, -1],
				  [0, -1, 0],
				  ]
	img_src = cv2.imread(img)

	kernel = np.array(kernel)
	kernel = kernel

	img_rst = cv2.filter2D(img_src, -1, kernel)

	cv2.imwrite(f'result/{out_name}.jpg', img_rst)


if __name__ == "__main__":
	# 1. Sharpen kernel
	kernel = [[0, -1, 0],
			  [-1, 5, -1],
			  [0, -1, 0]]
	compute_kernel(kernel=kernel, out_name="1-sharpen_kernel")

	# 2. Laplacian kernel
	kernel = [[0, 1, 0],
			  [1, 4, 1],
			  [0, 1, 0]]
	compute_kernel(kernel=kernel, out_name="2-laplacian_kernel")

	# 3. Emboss kernel
	kernel = [[-2, -1, 0],
			  [-1, 1, 1],
			  [0, 1, 2]]
	compute_kernel(kernel=kernel, out_name="3-emboss_kernel")

	# 4. Outline kernel
	kernel = [[-1, -1, -1],
			  [-1, 9, -1],
			  [-1, -1, -1]]
	compute_kernel(kernel=kernel, out_name="4-outline_kernel")

	# 5. Bottom sobel
	kernel = [[-1, -2, -1],
			  [0, 0, 0],
			  [1, 2, 1]]
	compute_kernel(kernel=kernel, out_name="5-bottom_sobel")

	# 6. Right sobel
	kernel = [[-1, 0, 1],
			  [-2, 0, 2],
			  [-1, 0, 1]]
	compute_kernel(kernel=kernel, out_name="6-right_sobel")

	# 7. Top sobel
	kernel = [[1, 2, 1],
			  [0, 0, 0],
			  [-1, -2, -1]]
	compute_kernel(kernel=kernel, out_name="7-top_sobel")

	# 8. Outline kernel
	kernel = [[-1, -1, -1],
			  [-1, 8, -1],
			  [-1, -1, -1]]
	compute_kernel(kernel=kernel, out_name="8-difference_kernel")

	# 9. Weighted average
	kernel = [[1, 1, 1],
			  [1, 8, 1],
			  [1, 1, 1]]
	compute_kernel(kernel=kernel, out_name="9-weighted_average")
