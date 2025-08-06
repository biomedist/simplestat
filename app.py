import os
from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
from scipy import stats

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 파일 확장자 체크
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 라우트: 메인 페이지
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # 파일 업로드 처리
        file = request.files['file']
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            return redirect(url_for('analyze', filename=file.filename))
        else:
            return '지원되지 않는 파일 형식입니다.'
    return render_template('index.html')

# 라우트: 분석 결과
@app.route('/analyze/<filename>')
def analyze(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    df = pd.read_excel(filepath)

    # 간단한 가정: 두 컬럼이 있어야 함
    if df.shape[1] < 2:
        return '데이터에 두 개 이상의 컬럼이 필요합니다.'

    # 첫 두 컬럼만 사용
    col1, col2 = df.columns[:2]
    group1 = df[col1].dropna()
    group2 = df[col2].dropna()

    # 각 그룹의 통계량 계산
    mean1, mean2 = group1.mean(), group2.mean()
    std1, std2 = group1.std(ddof=1), group2.std(ddof=1)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

    # T-test 수행
    t_stat, p_value = stats.ttest_ind(group1, group2)
    # 자유도 계산 (Welch's t-test 기준)
    n1, n2 = len(group1), len(group2)
    s1_sq, s2_sq = var1, var2
    dfree = (s1_sq/n1 + s2_sq/n2)**2 / ( ((s1_sq/n1)**2)/(n1-1) + ((s2_sq/n2)**2)/(n2-1) )
    # Effect size (Cohen's d)
    pooled_std = (( (n1-1)*s1_sq + (n2-1)*s2_sq ) / (n1+n2-2))**0.5 if n1+n2-2 > 0 else 0
    cohen_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

    result = {
        'column1': col1,
        'column2': col2,
        'mean1': round(mean1, 4),
        'mean2': round(mean2, 4),
        'std1': round(std1, 4),
        'std2': round(std2, 4),
        'var1': round(var1, 4),
        'var2': round(var2, 4),
        't_statistic': round(t_stat, 4),
        'df': int(round(dfree)),
        'p_value': round(p_value, 4),
        'alpha': 0.05,
        'cohen_d': round(cohen_d, 4),
        'significant': p_value < 0.05
    }

    # 결과 해석 생성
    if p_value < 0.05:
        interpretation = f"p-값이 0.05보다 작으므로 두 그룹({col1}, {col2}) 간에 통계적으로 유의미한 차이가 있습니다."
    else:
        interpretation = f"p-값이 0.05 이상이므로 두 그룹({col1}, {col2}) 간에 통계적으로 유의미한 차이가 없습니다."

    # 업로드된 파일 및 메모리 캐시 삭제
    try:
        os.remove(filepath)
    except Exception:
        pass
    del df, group1, group2
    import gc
    gc.collect()

    return render_template('index.html', result=result, interpretation=interpretation, filename=filename)


