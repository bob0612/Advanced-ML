One-off helpers that ran on the RunPod servers (or locally) besides the scripts in `experiments/`.
The servers otherwise ran this repository's code (`server_setup.sh` clones it). None of these touch the test split for selection.
Report PDF: `pandoc report/report.md -s --css style.css --embed-resources -o report.html && node render_report_pdf.js report.html report.pdf`.
