"""
Generate presentation-ready HTML from evolution analytics.

SPRINT 14: HTML presentation generator for academic submission.

Creates self-contained HTML file with embedded images for easy sharing.
"""

import markdown
from pathlib import Path
import base64
import sys


def generate_html_presentation(report_path: str, images_dir: str, output_path: str):
    """
    Generate HTML presentation from markdown report and plots.

    Args:
        report_path: Path to evolution_report.md
        images_dir: Directory containing PNG plots
        output_path: Output HTML file path
    """
    report_path = Path(report_path)
    images_dir = Path(images_dir)
    output_path = Path(output_path)

    if not report_path.exists():
        print(f"[ERROR] Report not found: {report_path}")
        return False

    # Read markdown report
    with open(report_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # Convert to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code'])

    # Read and encode images
    def encode_image(path):
        """Encode image to base64 for embedding."""
        try:
            with open(path, 'rb') as f:
                return base64.b64encode(f.read()).decode()
        except FileNotFoundError:
            print(f"[WARNING] Image not found: {path}")
            return ""

    fitness_img = encode_image(images_dir / "fitness_evolution.png")
    performance_img = encode_image(images_dir / "performance_metrics.png")
    modules_img = encode_image(images_dir / "module_trends.png")

    # Build full HTML with embedded images
    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GA Evolution Report - Academic Presentation</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 50px;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        }}

        .header {{
            text-align: center;
            padding-bottom: 30px;
            border-bottom: 4px solid #667eea;
            margin-bottom: 40px;
        }}

        .header h1 {{
            color: #2c3e50;
            font-size: 2.5em;
            margin-bottom: 10px;
        }}

        .header p {{
            color: #7f8c8d;
            font-size: 1.1em;
        }}

        h1 {{
            color: #2c3e50;
            margin-top: 40px;
            margin-bottom: 20px;
            font-size: 2em;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
        }}

        h2 {{
            color: #34495e;
            margin-top: 30px;
            margin-bottom: 15px;
            font-size: 1.6em;
        }}

        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
            margin-bottom: 10px;
            font-size: 1.3em;
        }}

        p {{
            margin-bottom: 15px;
            text-align: justify;
        }}

        ul, ol {{
            margin-left: 30px;
            margin-bottom: 15px;
        }}

        li {{
            margin-bottom: 8px;
        }}

        .image-container {{
            margin: 30px 0;
            text-align: center;
        }}

        img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}

        img:hover {{
            transform: scale(1.02);
        }}

        code {{
            background-color: #f8f9fa;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: 'Courier New', Courier, monospace;
            font-size: 0.9em;
            color: #e74c3c;
        }}

        pre {{
            background-color: #2c3e50;
            color: #ecf0f1;
            padding: 20px;
            border-radius: 6px;
            overflow-x: auto;
            margin: 20px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}

        th {{
            background-color: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}

        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
        }}

        tr:hover {{
            background-color: #f8f9fa;
        }}

        .metric-box {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 25px;
            border-radius: 8px;
            margin: 20px 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }}

        .metric-box h3 {{
            color: white;
            margin-top: 0;
        }}

        .metric-box ul {{
            margin-left: 20px;
        }}

        .metric-box li {{
            color: #ecf0f1;
        }}

        .footer {{
            margin-top: 50px;
            padding-top: 30px;
            border-top: 2px solid #e0e0e0;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}

        .badge {{
            display: inline-block;
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: 600;
            margin: 5px;
        }}

        .badge-success {{
            background-color: #27ae60;
            color: white;
        }}

        .badge-warning {{
            background-color: #f39c12;
            color: white;
        }}

        .badge-danger {{
            background-color: #e74c3c;
            color: white;
        }}

        .badge-info {{
            background-color: #3498db;
            color: white;
        }}

        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 30px 0;
        }}

        @media print {{
            body {{
                background: white;
                padding: 0;
            }}

            .container {{
                box-shadow: none;
                padding: 20px;
            }}

            img {{
                page-break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>[GA] Genetic Algorithm Evolution Report</h1>
            <p>Cryptocurrency Trading Pattern Discovery</p>
            <p><strong>SPRINT 14 - Enhanced Analytics</strong></p>
        </div>

        {html_content}

        <hr>

        <h2>[DATA] Visualizations</h2>

        <div class="image-container">
            <h3>Fitness Evolution Over Generations</h3>
            <img src="data:image/png;base64,{fitness_img}" alt="Fitness Evolution">
            <p><em>Shows how the population's best, mean, and median fitness evolved over generations, with standard deviation bands.</em></p>
        </div>

        <div class="image-container">
            <h3>Performance Metrics Evolution</h3>
            <img src="data:image/png;base64,{performance_img}" alt="Performance Metrics">
            <p><em>Tracks average Sharpe Ratio, CAGR, and trade frequency across the population over time.</em></p>
        </div>

        <div class="image-container">
            <h3>Module Usage Trends</h3>
            <img src="data:image/png;base64,{modules_img}" alt="Module Trends">
            <p><em>Illustrates which trading pattern modules (building blocks) became dominant as evolution progressed.</em></p>
        </div>

        <div class="footer">
            <p><strong>Generated by Evolution Analytics System</strong></p>
            <p>Universidad del CEMA (UCEMA) - Advanced Business Analytics</p>
            <p>Genetic Algorithm for Crypto Trading Pattern Discovery</p>
        </div>
    </div>
</body>
</html>
"""

    # Write HTML
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_template)

    print(f"[OK] Generated presentation: {output_path}")
    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate HTML presentation from evolution analytics')
    parser.add_argument('--report', type=str, default="./analysis_output/evolution_report.md",
                       help='Path to markdown report')
    parser.add_argument('--images', type=str, default="./analysis_output",
                       help='Directory containing PNG plots')
    parser.add_argument('--output', type=str, default="./analysis_output/presentation.html",
                       help='Output HTML file path')

    args = parser.parse_args()

    success = generate_html_presentation(
        report_path=args.report,
        images_dir=args.images,
        output_path=args.output
    )

    if success:
        print("")
        print("="*80)
        print("[OK] PRESENTATION READY!")
        print("="*80)
        print(f"\nOpen in browser: {Path(args.output).absolute()}")
        print("\nYou can now share this self-contained HTML file with your professor.")
        print("="*80)
        print("")
    else:
        print("\n[ERROR] Presentation generation failed")
        sys.exit(1)
