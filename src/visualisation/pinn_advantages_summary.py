import matplotlib.pyplot as plt
import numpy as np
import os

def create_summary_visualization():
    """Create a summary visualization of PINN advantages over CFD"""
    # Use a simple style that's guaranteed to be available
    plt.style.use('default')
    
    # Set figure-wide parameters
    plt.rcParams.update({
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': (15, 10),
        'figure.dpi': 100,
    })
    
    fig = plt.figure()
    
    # 1. Solution Density Comparison (Bar Plot)
    ax1 = plt.subplot(221)
    densities = [118, 10721.5]  # In millions points/m3
    labels = ['CFD', 'PINN']
    colors = ['#AED6F1', '#A2D9A2']  # Softer blue and green
    ax1.bar(labels, densities, color=colors)
    ax1.set_title('Solution Point Density', pad=10)
    ax1.set_ylabel('Million points/m3')
    # Add value labels on bars
    for i, v in enumerate(densities):
        ax1.text(i, v, f'{v:,.0f}M', ha='center', va='bottom')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # 2. Computational Time Comparison (Log Scale)
    ax2 = plt.subplot(222)
    methods = ['CFD Setup', 'CFD Mesh Gen', 'CFD Runtime', 'PINN Training', 'PINN Inference']
    times = [48, 24, 72, 12, 0.001]  # Times in hours
    colors = ['#AED6F1']*3 + ['#A2D9A2']*2
    bars = ax2.barh(methods, times, color=colors)
    ax2.set_title('Computational Time Comparison', pad=10)
    ax2.set_xscale('log')
    ax2.set_xlabel('Hours (log scale)')
    ax2.grid(True, linestyle='--', alpha=0.3)
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if times[i] < 1:
            label = f'{times[i]*3600:.0f} sec'
        else:
            label = f'{times[i]:.0f} hrs'
        ax2.text(width, bar.get_y() + bar.get_height()/2, 
                f'  {label}', va='center')
    
    # 3. Scaling Performance (Line Plot)
    ax3 = plt.subplot(223)
    points = [100, 1000, 10000]
    throughput = [2580, 18726, 104274]  # Points per second
    ax3.plot(points, throughput, '-o', color='#2ECC71', linewidth=2, 
             markersize=8, markerfacecolor='white')
    ax3.set_title('PINN Scaling Performance', pad=10)
    ax3.set_xlabel('Sample Size (points)')
    ax3.set_ylabel('Points Processed per Second')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.grid(True, linestyle='--', alpha=0.3)
    # Add value labels
    for x, y in zip(points, throughput):
        ax3.text(x, y*1.1, f'{y:,.0f}', ha='center', va='bottom')
    
    # 4. Feature Comparison (Table)
    ax4 = plt.subplot(224)
    ax4.axis('off')
    features = [
        ['Feature', 'CFD', 'PINN'],
        ['Mesh Required', 'Yes', 'No'],
        ['Solution Space', 'Discrete', 'Continuous'],
        ['Real-time Capable', 'No', 'Yes'],
        ['Parameter Exploration', 'Slow', 'Fast'],
        ['Memory Usage', 'High', 'Low'],
        ['Training Required', 'No', 'Yes'],
        ['Physics Constraints', 'Implicit', 'Explicit']
    ]
    table = ax4.table(cellText=features, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color the header and cells
    for i in range(len(features)):
        for j in range(len(features[0])):
            cell = table._cells[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#F2F4F4')
            elif j > 0:  # Values
                if features[i][j] in ['Yes', 'Continuous', 'Fast', 'Low', 'Explicit']:
                    cell.set_facecolor('#A2D9A2')  # Soft green
                elif features[i][j] in ['No', 'Discrete', 'Slow', 'High', 'Implicit']:
                    cell.set_facecolor('#AED6F1')  # Soft blue
    
    plt.suptitle('PINN vs CFD: Comparative Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Ensure the output directory exists
    os.makedirs('figures/pinn_analysis', exist_ok=True)
    
    try:
        # Save the visualization
        output_path = 'figures/pinn_analysis/pinn_advantages_summary.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Successfully saved visualization to {output_path}")
    except Exception as e:
        print(f"Error saving visualization: {str(e)}")
    finally:
        plt.close()

if __name__ == "__main__":
    print("Creating PINN vs CFD comparative analysis visualization...")
    try:
        create_summary_visualization()
    except Exception as e:
        print(f"Error creating visualization: {str(e)}")
