import os
from skimage import io, metrics
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



def evaluate_images(model_name, image_folder):
    results = []
    for scale_factor in ['x2', 'x4', 'x8']:
        if model_name == 'EDSR' and scale_factor == 'x8':
            continue
        model_path = os.path.join(image_folder, scale_factor)
        ground_truth_path = os.path.join(image_folder, "Ground_Truth")

        i = 0
        for image_name in os.listdir(model_path):

            
            model_image_path = os.path.join(model_path, image_name)
            ground_truth_image_path = os.path.join(ground_truth_path, os.listdir(ground_truth_path)[i])

            i += 1

            # Read images
            model_image = io.imread(model_image_path)
            ground_truth_image = io.imread(ground_truth_image_path)

            # Calculate metrics
            mse = metrics.mean_squared_error(ground_truth_image, model_image)
            psnr = metrics.peak_signal_noise_ratio(ground_truth_image, model_image)
            ssim = metrics.structural_similarity(ground_truth_image, model_image, channel_axis=False)

            results.append({
                'Model': model_name,
                'Scale': scale_factor,
                'Image': image_name,
                'MSE': mse,
                'PSNR': psnr,
                'SSIM': ssim
            })

    return results





def calculate_average_metrics_by_model(results):
    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Calculate average metrics for each model
    avg_metrics = df.groupby(['Model'])[['MSE', 'PSNR', 'SSIM']].mean().reset_index()

    return avg_metrics

def calculate_average_metrics_by_scale(results):
    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Calculate average metrics for each model and magnification
    avg_metrics = df.groupby(['Model', 'Scale'])[['MSE', 'PSNR', 'SSIM']].mean().reset_index()

    return avg_metrics

def plot_average_metrics_by_model(avg_metrics):
    # Set the style
    sns.set(style="whitegrid")

    metrics = ['MSE', 'PSNR', 'SSIM']

    for metric in metrics:
        # Create a new figure for each metric
        plt.figure(figsize=(8, 6))

        # Plot bar plot for the average metric
        sns.barplot(data=avg_metrics, x='Model', y=metric, palette="viridis")
        plt.ylabel(metric)
        plt.title(f'Average {metric} Comparison Across Models')

        # Annotate bars with average values
        for p in plt.gca().patches:
            plt.gca().annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                               ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        plt.xlabel('Model')

        plt.tight_layout()
        plt.show()


def plot_average_metrics_by_scale(avg_metrics):
    # Set the style
    sns.set(style="whitegrid")

    magnifications = ['x2', 'x4', 'x8']
    metrics = ['MSE', 'PSNR', 'SSIM']

    for magnification in magnifications:
        # Filter data for the current magnification
        magnification_data = avg_metrics[avg_metrics['Scale'] == magnification]

        # Create a new figure for each magnification
        plt.figure(figsize=(8, 8))
        
        for i, metric in enumerate(metrics, start=1):
            # Create subplots for each metric
            plt.subplot(1, 3, i)

            # Plot bar plot for the average metric
            sns.barplot(data=magnification_data, x='Model', y=metric, palette="viridis")
            plt.ylabel(metric)
            plt.title(f'Average {metric} Comparison for {magnification} Magnification', pad=20)

            # Annotate bars with average values
            for p in plt.gca().patches:
                plt.gca().annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                                   ha='center', va='center', xytext=(0, 10), textcoords='offset points')

        plt.xlabel('Model')

        plt.subplots_adjust(hspace=.5, top=1.5)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    srganResults = evaluate_images('SRGAN', '/Users/jaernizam/Desktop/MIPS/Final_Project/SRGAN_Results')
    edsrResults = evaluate_images('EDSR', '/Users/jaernizam/Desktop/MIPS/Final_Project/EDSR_Results')
    esrganResults = evaluate_images('Real-ESRGAN', '/Users/jaernizam/Desktop/MIPS/Final_Project/Real_ESRGAN_Results')

    all_results = srganResults + edsrResults + esrganResults
    
    avgModels = calculate_average_metrics_by_model(all_results)
    avgScales = calculate_average_metrics_by_scale(all_results)


    plot_average_metrics_by_model(avgModels)
    plot_average_metrics_by_scale(avgScales)

    