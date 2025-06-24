from flask import Flask, request, render_template_string, send_file
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import time
from program import jalanPCA, ambilJml, ambilJpgQ, ukurStr, cepetPCA, fig2b64

app = Flask(__name__)

img_ori = None
img_cmp = None
size_ori = 0
size_cmp = 0
buf_cmp = None


HTML_PAGE = '''
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image Compressor</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #6366f1; --secondary: #8b5cf6; --accent: #06b6d4;
            --success: #10b981; --warning: #f59e0b; --error: #ef4444;
            --bg-primary: #0f172a; --bg-secondary: #1e293b; --bg-tertiary: #334155;
            --surface: rgba(30, 41, 59, 0.8); --surface-light: rgba(51, 65, 85, 0.6);
            --text: #f8fafc; --text-secondary: #cbd5e1; --text-muted: #94a3b8;
            --border: rgba(148, 163, 184, 0.2); --shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            --glow: 0 0 20px rgba(99, 102, 241, 0.3);
        }
        
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: 'Inter', sans-serif; background: var(--bg-primary);
            color: var(--text); line-height: 1.6; min-height: 100vh;
            background-image: 
                radial-gradient(circle at 20% 80%, rgba(99, 102, 241, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(139, 92, 246, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(6, 182, 212, 0.05) 0%, transparent 50%);
        }
        
        .animated-bg {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1;
            background: linear-gradient(-45deg, #0f172a, #1e293b, #334155, #1e293b);
            background-size: 400% 400%; animation: gradientShift 15s ease infinite;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        /* About Us Button */
        .about-btn {
            position: fixed;
            top: 2rem;
            right: 2rem;
            z-index: 1000;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 50px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-size: 0.9rem;
        }
        
        .about-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 12px 35px rgba(99, 102, 241, 0.4);
        }
        
        /* Modal Styles */
        .modal {
            display: none;
            position: fixed;
            z-index: 2000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            animation: fadeIn 0.3s ease;
        }
        
        .modal.show {
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .modal-content {
            background: var(--surface);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border);
            border-radius: 20px;
            padding: 2rem;
            max-width: 800px;
            width: 90%;
            max-height: 90vh;
            overflow-y: auto;
            position: relative;
            box-shadow: var(--shadow);
            animation: slideIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from { transform: translateY(-50px) scale(0.9); opacity: 0; }
            to { transform: translateY(0) scale(1); opacity: 1; }
        }
        
        .modal-header {
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border);
        }
        
        .modal-title {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 0.5rem;
        }
        
        .modal-subtitle {
            color: var(--text-secondary);
            font-size: 1.1rem;
            font-weight: 500;
        }
        
        .close {
            position: absolute;
            top: 1rem;
            right: 1.5rem;
            color: var(--text-muted);
            font-size: 2rem;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            width: 40px;
            height: 40px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 50%;
            background: var(--surface-light);
        }
        
        .close:hover {
            color: var(--text);
            background: var(--bg-tertiary);
            transform: scale(1.1);
        }
        
        .team-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.5rem;
            margin-top: 2rem;
        }
        
        .team-member {
            background: var(--surface-light);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .team-member::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
        }
        
        .team-member:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.2), var(--glow);
        }
        
        .member-photo {
            width: 80px;
            height: 80px;
            border-radius: 50%;
            margin: 0 auto 1rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2rem;
            color: white;
            font-weight: bold;
            box-shadow: 0 8px 20px rgba(99, 102, 241, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .member-photo img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 50%;
        }
        
        .member-name {
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text);
            margin-bottom: 0.5rem;
        }
        
        .member-id {
            font-size: 0.85rem;
            color: var(--accent);
            font-weight: 500;
            margin-bottom: 0.75rem;
            background: rgba(6, 182, 212, 0.1);
            padding: 0.25rem 0.75rem;
            border-radius: 20px;
            display: inline-block;
        }
        
        .member-role {
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-style: italic;
        }
        
        .container { max-width: 1200px; margin: 0 auto; padding: 2rem 1rem; }
        
        .header {
            text-align: center; margin-bottom: 3rem; animation: fadeInUp 0.8s ease-out;
            margin-top: 2rem;
        }
        
        .header h1 {
            font-size: clamp(2.5rem, 5vw, 4rem); font-weight: 700; margin-bottom: 1rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary), var(--accent));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
            text-shadow: 0 0 30px rgba(99, 102, 241, 0.3);
        }
        
        .header p { 
            font-size: 1.2rem; color: var(--text-secondary); font-weight: 300; 
            max-width: 600px; margin: 0 auto;
        }
        
        .glass-card {
            background: var(--surface); backdrop-filter: blur(20px); border-radius: 20px; 
            padding: 2rem; margin-bottom: 2rem; border: 1px solid var(--border);
            box-shadow: var(--shadow); animation: fadeInUp 0.6s ease-out;
            position: relative; overflow: hidden;
        }
        
        .glass-card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
            background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.5), transparent);
        }
        
        @keyframes fadeInUp { 
            from { opacity: 0; transform: translateY(30px); } 
            to { opacity: 1; transform: translateY(0); } 
        }
        
        .grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); 
            gap: 2rem; margin-bottom: 2rem; 
        }
        
        .form-group { display: flex; flex-direction: column; gap: 0.75rem; }
        
        .label {
            font-weight: 600; color: var(--text); font-size: 0.95rem;
            display: flex; align-items: center; gap: 0.75rem;
        }
        
        .upload-area {
            border: 2px dashed var(--border); border-radius: 16px; padding: 3rem 2rem; 
            text-align: center; cursor: pointer; background: var(--surface-light);
            transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1); position: relative;
            backdrop-filter: blur(10px);
        }
        
        .upload-area:hover {
            border-color: var(--primary); background: rgba(99, 102, 241, 0.1);
            transform: translateY(-4px); box-shadow: var(--glow);
        }
        
        .upload-icon { 
            font-size: 3rem; margin-bottom: 1rem; 
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        
        .upload-text { color: var(--text-secondary); font-weight: 500; font-size: 1.1rem; }
        .file-name { margin-top: 1rem; font-size: 0.9rem; color: var(--accent); font-weight: 600; }
        
        .slider-container { position: relative; margin-top: 1rem; }
        
        .slider {
            width: 100%; height: 8px; border-radius: 4px; background: var(--bg-tertiary);
            outline: none; -webkit-appearance: none; cursor: pointer;
            background: linear-gradient(to right, var(--success) 0%, var(--warning) 50%, var(--error) 100%);
        }
        
        .slider::-webkit-slider-thumb {
            -webkit-appearance: none; width: 24px; height: 24px; border-radius: 50%;
            background: linear-gradient(135deg, var(--primary), var(--secondary)); 
            cursor: pointer; box-shadow: 0 4px 12px rgba(99, 102, 241, 0.4);
            transition: all 0.3s ease; border: 2px solid var(--text);
        }
        
        .slider::-webkit-slider-thumb:hover { 
            transform: scale(1.2); box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6); 
        }
        
        .slider-value {
            position: absolute; top: -45px; right: 0; 
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white; padding: 0.5rem 0.75rem; border-radius: 8px; 
            font-size: 0.85rem; font-weight: 600; box-shadow: var(--glow);
        }
        
        .quality-indicator {
            display: flex; justify-content: space-between; margin-top: 0.5rem;
            font-size: 0.8rem; color: var(--text-muted);
        }
        
        .preset-buttons {
            display: flex; gap: 0.5rem; margin-top: 1rem;
        }
        
        .preset-btn {
            flex: 1; padding: 0.5rem; border: 1px solid var(--border); 
            border-radius: 8px; background: var(--surface-light); color: var(--text);
            cursor: pointer; transition: all 0.3s ease; font-size: 0.8rem;
        }
        
        .preset-btn:hover, .preset-btn.active {
            border-color: var(--primary); background: rgba(99, 102, 241, 0.1);
        }
        
        .input {
            width: 100%; padding: 1rem 1.25rem; border: 1px solid var(--border); 
            border-radius: 12px; font-size: 1rem; background: var(--surface-light);
            color: var(--text); transition: all 0.3s ease; backdrop-filter: blur(10px);
        }
        
        .input:focus { 
            outline: none; border-color: var(--primary); 
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2), var(--glow);
        }
        
        .btn {
            width: 100%; padding: 1rem 2rem; border: none; border-radius: 12px; 
            font-size: 1rem; font-weight: 600; cursor: pointer; 
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            display: flex; align-items: center; justify-content: center; gap: 0.75rem;
            position: relative; overflow: hidden;
        }
        
        .btn::before {
            content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.5s ease;
        }
        
        .btn:hover::before { left: 100%; }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: white; box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
        }
        
        .btn-primary:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 12px 35px rgba(99, 102, 241, 0.4); 
        }
        
        .btn-secondary {
            background: linear-gradient(135deg, var(--accent), var(--success));
            color: white; box-shadow: 0 8px 25px rgba(6, 182, 212, 0.3);
        }
        
        .btn-secondary:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 12px 35px rgba(6, 182, 212, 0.4); 
        }
        
        .section-title {
            font-size: 1.75rem; font-weight: 700; color: var(--text); margin-bottom: 2rem;
            display: flex; align-items: center; gap: 1rem; position: relative;
        }
        
        .section-title::before {
            content: ''; width: 4px; height: 40px; border-radius: 2px;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            box-shadow: var(--glow);
        }
        
        .image-grid { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); 
            gap: 2rem; margin-bottom: 2rem; 
        }
        
        .image-card {
            background: var(--surface); backdrop-filter: blur(20px); border: 1px solid var(--border);
            border-radius: 16px; overflow: hidden; transition: all 0.4s ease;
            box-shadow: var(--shadow);
        }
        
        .image-card:hover { 
            transform: translateY(-8px) scale(1.02); 
            box-shadow: 0 30px 60px rgba(0, 0, 0, 0.3), var(--glow);
        }
        
        .image-header {
            padding: 1.5rem; background: linear-gradient(135deg, var(--surface), var(--surface-light));
            border-bottom: 1px solid var(--border); font-weight: 600; text-align: center;
            color: var(--text); font-size: 1.1rem;
        }
        
        .image-body { padding: 1.5rem; }
        .image-display { width: 100%; height: auto; border-radius: 12px; }
        
        .file-info {
            margin-top: 1rem; text-align: center; font-size: 0.95rem; 
            color: var(--accent); font-weight: 600; 
            background: rgba(6, 182, 212, 0.1); padding: 0.75rem; 
            border-radius: 8px; border: 1px solid rgba(6, 182, 212, 0.2);
        }
        
        .metrics { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); 
            gap: 1.5rem; margin-bottom: 2rem; 
        }
        
        .metric {
            background: var(--surface); backdrop-filter: blur(20px); border: 1px solid var(--border);
            border-radius: 16px; padding: 2rem; text-align: center; 
            transition: all 0.4s ease; box-shadow: var(--shadow); position: relative;
        }
        
        .metric::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, var(--primary), var(--secondary), var(--accent));
            border-radius: 16px 16px 0 0;
        }
        
        .metric:hover { 
            transform: translateY(-4px); 
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2), var(--glow);
        }
        
        .metric-value { 
            font-size: 2rem; font-weight: 700; margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        
        .metric-label { color: var(--text-secondary); font-size: 0.85rem; font-weight: 500; }
        
        .warning {
            background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(245, 158, 11, 0.05));
            border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; 
            padding: 1.5rem; margin-bottom: 2rem; color: var(--warning);
            backdrop-filter: blur(10px);
        }
        
        .warning h4 { color: var(--warning); font-weight: 600; margin-bottom: 0.5rem; }
        
        .success {
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(16, 185, 129, 0.05));
            border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; 
            padding: 1.5rem; margin-bottom: 2rem; color: var(--success);
            backdrop-filter: blur(10px);
        }
        
        .success h4 { color: var(--success); font-weight: 600; margin-bottom: 0.5rem; }
        
        .charts { 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); 
            gap: 2rem; margin-bottom: 2rem; 
        }
        
        .chart {
            background: var(--surface); backdrop-filter: blur(20px); border: 1px solid var(--border);
            border-radius: 16px; padding: 2rem; transition: all 0.4s ease; 
            box-shadow: var(--shadow);
        }
        
        .chart:hover { 
            transform: translateY(-4px); 
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2), var(--glow);
        }
        
        .chart-title { 
            font-size: 1.1rem; font-weight: 600; color: var(--text); 
            margin-bottom: 1.5rem; text-align: center; 
        }
        
        .chart-img { width: 100%; height: auto; border-radius: 12px; }
        
        .download { text-align: center; margin-top: 2rem; }
        
        .btn-download {
            display: inline-flex; align-items: center; gap: 0.75rem; 
            background: linear-gradient(135deg, var(--success), var(--accent));
            color: white; padding: 1rem 2rem; border-radius: 12px; 
            font-weight: 600; text-decoration: none; border: none; cursor: pointer;
            transition: all 0.3s ease; box-shadow: 0 8px 25px rgba(16, 185, 129, 0.3);
            position: relative; overflow: hidden;
        }
        
        .btn-download::before {
            content: ''; position: absolute; top: 0; left: -100%; width: 100%; height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
            transition: left 0.5s ease;
        }
        
        .btn-download:hover::before { left: 100%; }
        .btn-download:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 12px 35px rgba(16, 185, 129, 0.4); 
        }
        
        .progress-container {
            display: none; margin: 2rem 0; text-align: center;
        }
        
        .progress-bar {
            width: 100%; height: 8px; background: var(--bg-tertiary); 
            border-radius: 4px; overflow: hidden; margin: 1rem 0;
        }
        
        .progress-fill {
            height: 100%; background: linear-gradient(90deg, var(--primary), var(--secondary));
            width: 0%; transition: width 0.3s ease; border-radius: 4px;
        }
        
        .progress-text {
            color: var(--text-secondary); font-size: 0.9rem; margin-bottom: 0.5rem;
        }
        
        .spinner {
            display: none; margin: 3rem auto; width: 50px; height: 50px; 
            border: 3px solid var(--border); border-radius: 50%; 
            border-top-color: var(--primary); animation: spin 1s linear infinite;
            box-shadow: var(--glow);
        }
        
        @keyframes spin { to { transform: rotate(360deg); } }
        
        .small { font-size: 0.85rem; color: var(--text-muted); margin-top: 0.5rem; }
        input[type="file"] { display: none; }
        
        .badge {
            display: inline-block; background: linear-gradient(135deg, var(--accent), var(--success));
            color: white; padding: 0.25rem 0.75rem; border-radius: 6px; 
            font-size: 0.75rem; font-weight: 600; margin-left: 0.75rem;
            box-shadow: 0 2px 8px rgba(6, 182, 212, 0.3);
        }
        
        @media (max-width: 768px) {
            .container { padding: 1rem; } 
            .glass-card { padding: 1.5rem; } 
            .header h1 { font-size: 2.5rem; }
            .grid, .image-grid, .charts { grid-template-columns: 1fr; } 
            .metrics { grid-template-columns: repeat(2, 1fr); }
            .about-btn {
                top: 1rem;
                right: 1rem;
                padding: 0.5rem 1rem;
                font-size: 0.8rem;
            }
            .modal-content {
                width: 95%;
                padding: 1.5rem;
            }
            .team-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="animated-bg"></div>
    
    <!-- About Us Button -->
    <button class="about-btn" onclick="openModal()">
         About Us
    </button>
    
    <!-- About Us Modal -->
    <div id="aboutModal" class="modal">
        <div class="modal-content">
            <span class="close" onclick="closeModal()">&times;</span>
            <div class="modal-header">
                <h2 class="modal-title">Kelompok 3</h2>
            </div>
            <div class="team-grid">
                <div class="team-member">
                    <div class="member-photo">
                        <img src="static/img/ayu.jpg" alt="Ayu">
                    </div>
                    <div class="member-name">Ayu Saniatus Sholihah</div>
                    <div class="member-id">L0124005</div>
                    <div class="member-role">Penulis Laporan & Pengembang Program</div>
                </div>
                <div class="team-member">
                    <div class="member-photo">
                        <img src="static/img/fadhil.jpg" alt="Fadhil">
                        
                    </div>
                    <div class="member-name">Fadhil Rusadi</div>
                    <div class="member-id">L0124013</div>
                    <div class="member-role">Penulis Laporan & Editor Video</div>
                </div>
                <div class="team-member">
                    <div class="member-photo">
                        <img src="static/img/nabil.jpg" alt="Nabil">
                    </div>
                    <div class="member-name">Muhamad Nabil Fannani</div>
                    <div class="member-id">L0124135</div>
                    <div class="member-role">Pengembang Program & GUI</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="header">
            <h1>Image Compressor</h1>
        </div>
        
        <div class="glass-card">
            <form method="post" enctype="multipart/form-data" id="form">
                <div class="grid">
                    <div class="form-group">
                        <label class="label">
                            Pilih Gambar
                            {% if uploaded_image %}<span class="badge">TERSIMPAN</span>{% endif %}
                        </label>
                        <label class="upload-area" for="file">
                            <div class="upload-icon">📁</div>
                            <div class="upload-text">Klik atau drag gambar ke sini</div>
                            <div class="file-name" id="fileName">
                                {% if uploaded_image %}Gambar siap diproses{% else %}Belum ada file dipilih{% endif %}
                            </div>
                        </label>
                        <input type="file" name="image" id="file" accept="image/*" {% if not uploaded_image %}required{% endif %}>
                    </div>
                    <div class="form-group">
                        <label class="label">Kualitas Kompresi</label>
                        <div class="slider-container">
                            <input type="range" name="quality" id="quality" min="10" max="90" value="50" class="slider" oninput="updateQualityValue(this.value)">
                            <div class="slider-value" id="qualityValue">50%</div>
                        </div>
                        <div class="quality-indicator">
                            <span style="color: var(--success);">Kompresi Tinggi</span>
                            <span style="color: var(--warning);">Sedang</span>
                            <span style="color: var(--error);">Kompresi Rendah</span>
                        </div>
                        <div class="preset-buttons">
                            <button type="button" class="preset-btn" onclick="setQuality(20)">Maksimal</button>
                            <button type="button" class="preset-btn" onclick="setQuality(50)">Sedang</button>
                            <button type="button" class="preset-btn" onclick="setQuality(80)">Minimal</button>
                        </div>
                        <div class="small">Kualitas rendah = ukuran file lebih kecil</div>
                    </div>
                </div>

                {% if uploaded_image %}
                <button type="submit" class="btn btn-secondary"> Proses Ulang</button>
                {% else %}
                <button type="submit" class="btn btn-primary"> Mulai Kompresi</button>
                {% endif %}
            </form>
        </div>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-text" id="progressText">Memproses gambar...</div>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
        </div>
        
        <div class="spinner" id="spinner"></div>
        
        {% if image %}
        <div id="results">
            {% if compression_success %}
            <div class="success">
                <h4> Kompresi Berhasil!</h4>
                <p>Ukuran file berhasil dikurangi dari {{ original_size }} menjadi {{ compressed_size }} ({{ file_reduction }}% lebih kecil)</p>
            </div>
            {% elif size_warning %}
            <div class="warning">
                <h4> Perhatian</h4>
                <p>Hasil kompresi ({{ compressed_size }}) lebih besar dari asli ({{ original_size }}). Coba kurangi kualitas atau resolusi.</p>
            </div>
            {% endif %}
            
            <div class="glass-card">
                <h2 class="section-title">🎯 Hasil Kompresi</h2>
                <div class="image-grid">
                    <div class="image-card">
                        <div class="image-header">Gambar Asli</div>
                        <div class="image-body">
                            <img src="data:image/png;base64,{{ original }}" alt="Original" class="image-display">
                            <div class="file-info">{{ original_size }}</div>
                        </div>
                    </div>
                    <div class="image-card">
                        <div class="image-header">Hasil Kompresi</div>
                        <div class="image-body">
                            <img src="data:image/{{ format }};base64,{{ image }}" alt="Compressed" class="image-display">
                            <div class="file-info">{{ compressed_size }}</div>
                        </div>
                    </div>
                </div>
                
                <h2 class="section-title">📊 Statistik Kompresi</h2>
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">{{ runtime }}s</div>
                        <div class="metric-label">Waktu Proses</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ varian }}%</div>
                        <div class="metric-label">Variansi Dipertahankan</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ file_reduction }}%</div>
                        <div class="metric-label">Pengurangan Ukuran</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">{{ components_used }}</div>
                        <div class="metric-label">Komponen PCA</div>
                    </div>
                </div>
                
                <div class="download">
                    <form action="/download">
                        <button type="submit" class="btn-download"> Unduh Hasil</button>
                    </form>
                </div>
            </div>
            
            <div class="glass-card">
                <h2 class="section-title">📈 Analisis Detail</h2>
                <div class="charts">
                    <div class="chart">
                        <h3 class="chart-title">Distribusi Variansi RGB</h3>
                        <img src="data:image/png;base64,{{ bar_chart }}" alt="RGB Chart" class="chart-img">
                    </div>
                    <div class="chart">
                        <h3 class="chart-title">Akumulasi Variansi (Channel Red)</h3>
                        <img src="data:image/png;base64,{{ cum_chart }}" alt="Cumulative Chart" class="chart-img">
                    </div>
                </div>
            </div>
        </div>
        {% endif %}
    </div>
    
    <script>
        // Modal functions
        function openModal() {
            document.getElementById('aboutModal').classList.add('show');
            document.body.style.overflow = 'hidden';
        }
        
        function closeModal() {
            document.getElementById('aboutModal').classList.remove('show');
            document.body.style.overflow = 'auto';
        }
        
        // Close modal when clicking outside
        window.onclick = function(event) {
            const modal = document.getElementById('aboutModal');
            if (event.target === modal) {
                closeModal();
            }
        }
        
        // Close modal with Escape key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });
        
        function updateQualityValue(v) {
            document.getElementById('qualityValue').textContent = v + '%';
            // Update active preset button
            document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
        }
        
        function setQuality(value) {
            document.getElementById('quality').value = value;
            updateQualityValue(value);
            // Update active preset button
            document.querySelectorAll('.preset-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
        }
        
        document.getElementById('file').addEventListener('change', function(e) {
            document.getElementById('fileName').textContent = e.target.files[0] ? e.target.files[0].name : 'Belum ada file dipilih';
        });
        
        document.getElementById('form').addEventListener('submit', function() {
            document.getElementById('progressContainer').style.display = 'block';
            document.getElementById('spinner').style.display = 'block';
            const results = document.getElementById('results');
            if (results) results.style.display = 'none';
            
            // Simulate progress
            let progress = 0;
            const progressFill = document.getElementById('progressFill');
            const progressText = document.getElementById('progressText');
            
            const interval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress > 90) progress = 90;
                progressFill.style.width = progress + '%';
                
                if (progress < 30) {
                    progressText.textContent = 'Memuat gambar...';
                } else if (progress < 60) {
                    progressText.textContent = 'Menerapkan PCA...';
                } else if (progress < 90) {
                    progressText.textContent = 'Mengoptimalkan format...';
                } else {
                    progressText.textContent = 'Menyelesaikan...';
                }
            }, 200);
            
            // Clear interval after form submission
            setTimeout(() => {
                clearInterval(interval);
                progressFill.style.width = '100%';
                progressText.textContent = 'Selesai!';
            }, 3000);
        });
        
        window.addEventListener('load', function() {
            {% if image %}
            document.getElementById('results').style.display = 'block';
            document.getElementById('progressContainer').style.display = 'none';
            document.getElementById('spinner').style.display = 'none';
            {% endif %}
            updateQualityValue(50);
        });
    </script>
</body>
</html>
'''


@app.route("/", methods=["GET", "POST"])
def index():
    global img_ori, img_cmp, size_ori, size_cmp, buf_cmp

    if request.method == "POST":
        if 'image' in request.files and request.files['image'].filename:
            f = request.files["image"]
            isi = f.read()
            size_ori = len(isi)
            f = io.BytesIO(isi)
            g = Image.open(f).convert("RGB")
            img_ori = g.copy()
        elif img_ori is not None:
            g = img_ori.copy()
        else:
            return "No image uploaded", 400
        
        q = int(request.form.get("quality", 50))
        mx = int(request.form.get("max_size", 500))
        d = int(request.form.get("max_dimension", 600))

        t0 = time.time()
        res = cepetPCA(g, q, mx, d)
        waktu = round(time.time() - t0, 3)

        c = res['img']
        v = res['var']
        e = res['eig']
        j = res['cmp']
        size_cmp = res['byt']
        fmt = res['fmt'].lower()
        buf_cmp = res['buf']
        img_cmp = Image.fromarray(c)

        rata = round(np.mean(v) * 100, 2)
        fr = round(100 - (size_cmp / size_ori * 100), 2)
        cek = size_cmp < size_ori
        peringatan = size_cmp >= size_ori

        plt.style.use('dark_background')
        fig1, ax1 = plt.subplots(figsize=(7, 5), facecolor='#1e293b')
        fig1.subplots_adjust(top=0.82)  # Tambahkan margin atas
        ax1.set_facecolor('#1e293b')

        warna = ['#6366f1', '#8b5cf6', '#06b6d4']
        bars = ax1.bar(['Red', 'Green', 'Blue'], [i * 100 for i in v], color=warna, alpha=0.8)

        ax1.set_ylabel("Persentase Variansi (%)", color='#f8fafc')
        ax1.set_title("Distribusi Variansi Channel RGB", color='#f8fafc', fontweight='bold', fontsize=13)
        ax1.set_ylim(0, 100)
        ax1.grid(True, alpha=0.2, color='#475569')
        ax1.tick_params(colors='#cbd5e1')

        for b, x in zip(bars, v):
            h = b.get_height()
            ax1.text(b.get_x() + b.get_width()/2., h - 5, f'{x*100:.1f}%', ha='center', va='bottom', fontweight='bold', color='#f8fafc')

        chart1 = fig2b64(fig1)
        plt.close(fig1)

        if e is not None and len(e) > 0:
            fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor='#1e293b')
            ax2.set_facecolor('#1e293b')
            kum = np.cumsum(e) / np.sum(e)
            ax2.plot(range(1, len(kum) + 1), kum, color='#6366f1', linewidth=3, marker='o', markersize=3)
            ax2.set_xlabel("Jumlah Komponen", color='#f8fafc')
            ax2.set_ylabel("Kumulatif Variansi", color='#f8fafc')
            ax2.set_title("Cumulative Explained Variance (Channel Red)", color='#f8fafc', fontweight='bold')
            ax2.grid(True, alpha=0.2, color='#475569')
            ax2.tick_params(colors='#cbd5e1')
            ax2.axvline(x=j, color='#06b6d4', linestyle='--', linewidth=2, label=f'Komponen Digunakan: {j}')
            ax2.legend(facecolor='#334155', edgecolor='#475569', labelcolor='#f8fafc')
            chart2 = fig2b64(fig2)
            plt.close(fig2)
        else:
            fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor='#1e293b')
            ax2.set_facecolor('#1e293b')
            ax2.text(0.5, 0.5, 'Data tidak tersedia', ha='center', va='center', transform=ax2.transAxes, color='#f8fafc')
            chart2 = fig2b64(fig2)
            plt.close(fig2)

        c64 = base64.b64encode(res['buf']).decode('utf-8')
        ori = io.BytesIO()
        img_ori.save(ori, format="PNG")
        ori64 = base64.b64encode(ori.getvalue()).decode('utf-8')

        return render_template_string(HTML_PAGE, image=c64, original=ori64, format=fmt,
                                   runtime=waktu, varian=rata, file_reduction=fr, components_used=j,
                                   original_size=ukurStr(size_ori), compressed_size=ukurStr(size_cmp),
                                   bar_chart=chart1, cum_chart=chart2, uploaded_image=True, 
                                   size_warning=peringatan, compression_success=cek)

    return render_template_string(HTML_PAGE, uploaded_image=img_ori is not None)



@app.route("/download")
def download():
    global buf_cmp
    if buf_cmp:
        b = io.BytesIO(buf_cmp)
        return send_file(b, mimetype='image/png', as_attachment=True, download_name='compressed_image.png')
    return "No compressed image available.", 404

if __name__ == "__main__":
    app.run(debug=True)