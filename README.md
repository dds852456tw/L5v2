# L5v2: 50 Startups 多元線性迴歸分析與純前端互動式平台

本專案實作 CRISP-DM 流程，針對 50 家新創公司的各項支出（研發、行政管理、行銷推廣與州別）進行多元線性迴歸模型預測，並提供 Python 訓練腳本與純前端單一檔案互動式視覺化網頁。

## 專案結構

- `main.py`: Python 版 CRISP-DM 機器學習流水線（使用 scikit-learn、pandas、matplotlib、joblib）
- `index.html`: 純前端單一檔案互動式網頁（Tailwind CSS 深色模式、math.js 常態方程式求解、PapaParse 動態解析、Chart.js 散佈圖與即時預測滑桿）
- `50_Startups.csv`: 50 家新創公司歷史資料集
- `test_predictions.csv`: 測試集模型預測結果與殘差
- `actual_vs_predicted.png`: 實際利潤 vs 預測利潤散佈圖
- `residual_plot.png`: 殘差分析圖
- `startup_profit_model.joblib`: 訓練完成之 scikit-learn Pipeline 封存檔
