import cv2
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.model.problem import Problem
from pymoo.optimize import minimize

def canny_edge_weight(image, sigma=1.0):
    edges = cv2.Canny(image.astype(np.uint8), 50, 150)
    weights = cv2.GaussianBlur(edges.astype(np.float64), (0, 0), sigma)
    weights /= np.max(weights)
    return weights

def evaluate_fusion(original, fused):
    cc = np.corrcoef(original.flatten(), fused.flatten())[0, 1]
    psnr = cv2.PSNR(original, fused)

    grad_original_x = cv2.Sobel(original, cv2.CV_64F, 1, 0, ksize=3)
    grad_original_y = cv2.Sobel(original, cv2.CV_64F, 0, 1, ksize=3)
    grad_fused_x = cv2.Sobel(fused, cv2.CV_64F, 1, 0, ksize=3)
    grad_fused_y = cv2.Sobel(fused, cv2.CV_64F, 0, 1, ksize=3)
    grad_original = np.sqrt(grad_original_x ** 2 + grad_original_y ** 2)
    grad_fused = np.sqrt(grad_fused_x ** 2 + grad_fused_y ** 2)
    grad_diff = np.abs(grad_original - grad_fused)
    avg_grad = np.mean(grad_diff)

    rmse = np.sqrt(np.mean((original - fused) ** 2))

    mean_original = np.mean(original)

    rase = rmse / mean_original

    return cc, psnr, avg_grad, rmse, rase

class ImageFusionProblem(Problem):
    def __init__(self, pan_image, ms_image, original_gray):
        super().__init__(n_var=2, n_obj=2, xl=np.zeros(2), xu=np.ones(2))
        self.pan_image = pan_image
        self.ms_image = ms_image
        self.original_gray = original_gray

    def _evaluate(self, x, out, *args, **kwargs):
        weight_pan, weight_ms = x

        fused_image = weight_pan * self.pan_image + weight_ms * self.ms_image

        cc, psnr, avg_grad, rmse, rase = evaluate_fusion(self.original_gray, fused_image)

        out["F"] = np.array([rmse, rase])

# 读取全色和多光谱图像
pan_image = cv2.imread('./data/fusion_images/PAN.png', cv2.IMREAD_GRAYSCALE)
ms_image = cv2.imread('./data/fusion_images/MS.png')

# HIS变换
hsv_image = cv2.cvtColor(ms_image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_image)

# 计算原始灰度图像
original_gray = cv2.cvtColor(ms_image, cv2.COLOR_BGR2GRAY)

# 定义图像融合问题
problem = ImageFusionProblem(pan_image, ms_image, original_gray)

# 定义NSGA2算法
algorithm = NSGA2(pop_size=100,
                  sampling=get_sampling("real_lhs"),
                  crossover=get_crossover("real_sbx", prob=0.9, eta=15),
                  mutation=get_mutation("real_pm", eta=20),
                  eliminate_duplicates=True)

# 运行优化
res = minimize(problem, algorithm)

# 获取 Pareto 最优解集
pareto_set = res.F

# 选择 Pareto 最优解中的权重
best_weights = pareto_set[np.argmin(pareto_set[:, 0])]  # 选择RMSE最小的权重

# 使用最优权重进行图像融合
fused_image = best_weights[0] * pan_image + best_weights[1] * ms_image

# 保存融合后的图像
output_path = f'./data/fusion_images/fused_image_nsga2.png'
cv2.imwrite(output_path, fused_image)
