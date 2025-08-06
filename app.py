import os
from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import numpy as np
import matplotlib.pyplot as plt # Matplotlib 임포트
import io # BytesIO 사용을 위해 임포트
import base64 # 이미지를 base64로 인코딩하기 위해 임포트

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'xlsx', 'csv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'your_super_secret_key_here_replace_with_a_strong_one' # 실제 배포 시에는 복잡하고 유추하기 어려운 값으로 변경하세요.

# 파일 확장자 체크 함수
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 메인 페이지 라우트: 파일 업로드 처리
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return '파일이 없습니다.'
        
        file = request.files['file']
        if file.filename == '':
            return '파일을 선택하세요.'
        
        if file and allowed_file(file.filename):
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            
            session['uploaded_filepath'] = filepath
            session['uploaded_filename'] = file.filename
            
            return redirect(url_for('show_analysis_options'))
        else:
            return '지원되지 않는 파일 형식입니다. .xlsx 또는 .csv 파일만 허용됩니다.'
    
    # GET 요청 (페이지 로드 또는 분석 초기화) 처리
    if 'uploaded_filepath' in session:
        try:
            os.remove(session['uploaded_filepath'])
        except Exception as e:
            print(f"Error deleting file: {e}")
        session.pop('uploaded_filepath', None)
        session.pop('uploaded_filename', None)
    
    return render_template('index.html', filename=None, result=None, interpretation=None, plot_url=None)

# 분석 옵션 선택 및 결과 표시 라우트
@app.route('/analysis_options', methods=['GET', 'POST'])
def show_analysis_options():
    if 'uploaded_filepath' not in session:
        return redirect(url_for('index'))

    filepath = session['uploaded_filepath']
    filename = session['uploaded_filename']
    result = None
    interpretation = None
    plot_url = None # 시각화 이미지 URL
    selected_method = request.form.get('analysis_method', 'independent') 

    try:
        if filename.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filename.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            return '지원되지 않는 파일 형식입니다.'

        if request.method == 'POST':
            analysis_method = request.form.get('analysis_method')
            selected_method = analysis_method

            # 모든 그룹 데이터 추출 및 정규성 검정 준비
            all_groups_data = []
            normality_results = {}
            for col in df.columns:
                group_data = df[col].dropna()
                if not group_data.empty:
                    all_groups_data.append(group_data)
                    # Shapiro-Wilk 정규성 검정 (p-value > 0.05 이면 정규성 만족)
                    shapiro_stat, shapiro_p = stats.shapiro(group_data)
                    normality_results[col] = {
                        'statistic': round(shapiro_stat, 4),
                        'p_value': round(shapiro_p, 4),
                        'normal': shapiro_p > 0.05
                    }

            # 시각화 (Box Plot) 생성
            plt.figure(figsize=(10, 6))
            # 데이터가 2개 이상인 컬럼만 선택하여 박스 플롯 생성
            plot_data = [df[col].dropna() for col in df.columns if not df[col].dropna().empty]
            plot_labels = [col for col in df.columns if not df[col].dropna().empty]

            if plot_data: # 데이터가 있을 때만 플롯 생성
                plt.boxplot(plot_data, labels=plot_labels)
                plt.title('데이터 분포 및 이상치 (Box Plot)', fontsize=14)
                plt.ylabel('값', fontsize=12)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Matplotlib 그래프를 BytesIO에 저장 후 base64 인코딩
                buf = io.BytesIO()
                plt.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plot_url = base64.b64encode(buf.read()).decode('utf-8')
                plt.close() # 플롯 닫기 (메모리 누수 방지)

            homoscedasticity_result = None
            
            if analysis_method == 'independent':
                if df.shape[1] < 2:
                    return '독립표본 t-test를 수행하려면 두 개 이상의 컬럼이 필요합니다.'
                
                col1, col2 = df.columns[0], df.columns[1]
                group1 = df[col1].dropna()
                group2 = df[col2].dropna()

                mean1, mean2 = group1.mean(), group2.mean()
                std1, std2 = group1.std(ddof=1), group2.std(ddof=1)
                var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

                # Levene's 등분산성 검정 (p-value > 0.05 이면 등분산성 만족)
                levene_stat, levene_p = stats.levene(group1, group2)
                homoscedasticity_result = {
                    'statistic': round(levene_stat, 4),
                    'p_value': round(levene_p, 4),
                    'equal_variance': levene_p > 0.05
                }
                
                # 등분산성 검정 결과에 따라 t-test 방법 선택
                # equal_var=True: 등분산성 만족 시 (Student's t-test)
                # equal_var=False: 등분산성 불만족 시 (Welch's t-test)
                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=homoscedasticity_result['equal_variance']) 
                
                n1, n2 = len(group1), len(group2)
                s1_sq, s2_sq = var1, var2

                # 자유도 계산 (등분산성 여부에 따라 다름)
                if homoscedasticity_result['equal_variance']: # 등분산성 만족 시
                    dfree = n1 + n2 - 2
                    pooled_std = np.sqrt(((n1 - 1) * s1_sq + (n2 - 1) * s2_sq) / dfree) if dfree > 0 else 0
                else: # 등분산성 불만족 시 (Welch's t-test)
                    if n1 > 1 and n2 > 1:
                        dfree_num = (s1_sq/n1 + s2_sq/n2)**2
                        dfree_den = (((s1_sq/n1)**2)/(n1-1)) + (((s2_sq/n2)**2)/(n2-1))
                        dfree = dfree_num / dfree_den
                    else:
                        dfree = 0
                    pooled_std = np.sqrt(s1_sq + s2_sq) # Welch's d는 조금 다름, 여기서는 간략화

                cohen_d = (mean1 - mean2) / pooled_std if pooled_std != 0 else 0

                result = {
                    'type': 'independent_t_test',
                    'column1': col1,
                    'column2': col2,
                    'mean1': round(mean1, 4),
                    'mean2': round(mean2, 4),
                    'std1': round(std1, 4),
                    'std2': round(std2, 4),
                    'var1': round(var1, 4),
                    'var2': round(var2, 4),
                    't_statistic': round(t_stat, 4),
                    'df': int(round(dfree)) if dfree > 0 else 0,
                    'p_value': round(p_value, 4),
                    'alpha': 0.05,
                    'cohen_d': round(cohen_d, 4),
                    'significant': p_value < 0.05,
                    'group_stats': {
                        col1: {'mean': round(mean1, 4), 'std': round(std1, 4), 'var': round(var1, 4)},
                        col2: {'mean': round(mean2, 4), 'std': round(std2, 4), 'var': round(var2, 4)}
                    },
                    'normality_results': normality_results,
                    'homoscedasticity_result': homoscedasticity_result
                }
                
                # 해석 생성
                interpretation_parts = []
                interpretation_parts.append(f"<h3>독립표본 t-test 결과 해석</h3>")
                
                # 정규성 해석
                normality_satisfied = True
                for col, res in normality_results.items():
                    if not res['normal']:
                        normality_satisfied = False
                        interpretation_parts.append(f"• 그룹 '{col}'의 정규성 검정 (Shapiro-Wilk): 통계량 = {res['statistic']}, p-값 = {res['p_value']}. p-값이 0.05 이하이므로 정규성 가정을 만족하지 않습니다.")
                    else:
                        interpretation_parts.append(f"• 그룹 '{col}'의 정규성 검정 (Shapiro-Wilk): 통계량 = {res['statistic']}, p-값 = {res['p_value']}. p-값이 0.05 초과이므로 정규성 가정을 만족합니다.")
                
                if not normality_satisfied:
                    interpretation_parts.append("정규성 가정이 만족되지 않았으므로, t-test 결과 해석에 유의해야 합니다. 데이터 변환이나 비모수 검정(예: Mann-Whitney U test)을 고려할 수 있습니다.")

                # 등분산성 해석
                if homoscedasticity_result['equal_variance']:
                    interpretation_parts.append(f"• 등분산성 검정 (Levene): 통계량 = {homoscedasticity_result['statistic']}, p-값 = {homoscedasticity_result['p_value']}. p-값이 0.05 초과이므로 등분산성 가정을 만족합니다. 따라서 Student's t-test가 적용되었습니다.")
                else:
                    interpretation_parts.append(f"• 등분산성 검정 (Levene): 통계량 = {homoscedasticity_result['statistic']}, p-값 = {homoscedasticity_result['p_value']}. p-값이 0.05 이하이므로 등분산성 가정을 만족하지 않습니다. 따라서 Welch's t-test가 적용되었습니다.")

                # t-test 결과 해석
                if p_value < 0.05:
                    interpretation_parts.append(f"• t-test 결과: t-값 = {round(t_stat, 4)}, 자유도 = {int(round(dfree))}, p-값 = {round(p_value, 4)}. p-값이 유의수준 0.05보다 작으므로, 두 그룹({col1}, {col2}) 간에는 통계적으로 유의미한 차이가 있습니다.")
                    if cohen_d:
                        interpretation_parts.append(f"• 효과 크기 (Cohen's d): {round(cohen_d, 4)}. 이는 두 그룹 간의 차이 크기를 나타냅니다.")
                else:
                    interpretation_parts.append(f"• t-test 결과: t-값 = {round(t_stat, 4)}, 자유도 = {int(round(dfree))}, p-값 = {round(p_value, 4)}. p-값이 유의수준 0.05 이상이므로, 두 그룹({col1}, {col2}) 간에는 통계적으로 유의미한 차이가 없습니다.")
                interpretation = "<br>".join(interpretation_parts)


            elif analysis_method == 'paired':
                if df.shape[1] < 2:
                    return '대응표본 t-test를 수행하려면 두 개 이상의 컬럼이 필요합니다.'
                
                col1, col2 = df.columns[0], df.columns[1]
                group1 = df[col1].dropna()
                group2 = df[col2].dropna()

                if len(group1) != len(group2):
                    return '대응표본 t-test는 두 그룹의 데이터 길이가 같아야 합니다.'

                mean1, mean2 = group1.mean(), group2.mean()
                std1, std2 = group1.std(ddof=1), group2.std(ddof=1)
                var1, var2 = group1.var(ddof=1), group2.var(ddof=1)

                t_stat, p_value = stats.ttest_rel(group1, group2)

                dfree = len(group1) - 1

                diff = group1 - group2
                cohen_d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else 0

                result = {
                    'type': 'paired_t_test',
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
                    'significant': p_value < 0.05,
                    'group_stats': {
                        col1: {'mean': round(mean1, 4), 'std': round(std1, 4), 'var': round(var1, 4)},
                        col2: {'mean': round(mean2, 4), 'std': round(std2, 4), 'var': round(var2, 4)}
                    },
                    'normality_results': normality_results # 대응표본도 각 그룹의 정규성 확인
                }
                
                # 해석 생성
                interpretation_parts = []
                interpretation_parts.append(f"<h3>대응표본 t-test 결과 해석</h3>")
                
                # 정규성 해석
                normality_satisfied = True
                for col, res in normality_results.items():
                    if not res['normal']:
                        normality_satisfied = False
                        interpretation_parts.append(f"• 그룹 '{col}'의 정규성 검정 (Shapiro-Wilk): 통계량 = {res['statistic']}, p-값 = {res['p_value']}. p-값이 0.05 이하이므로 정규성 가정을 만족하지 않습니다.")
                    else:
                        interpretation_parts.append(f"• 그룹 '{col}'의 정규성 검정 (Shapiro-Wilk): 통계량 = {res['statistic']}, p-값 = {res['p_value']}. p-값이 0.05 초과이므로 정규성 가정을 만족합니다.")
                
                if not normality_satisfied:
                    interpretation_parts.append("정규성 가정이 만족되지 않았으므로, t-test 결과 해석에 유의해야 합니다. 데이터 변환이나 비모수 검정(예: Wilcoxon signed-rank test)을 고려할 수 있습니다.")

                # t-test 결과 해석
                if p_value < 0.05:
                    interpretation_parts.append(f"• t-test 결과: t-값 = {round(t_stat, 4)}, 자유도 = {int(round(dfree))}, p-값 = {round(p_value, 4)}. p-값이 유의수준 0.05보다 작으므로, 두 그룹({col1}, {col2}) 간에는 통계적으로 유의미한 차이가 있습니다.")
                    if cohen_d:
                        interpretation_parts.append(f"• 효과 크기 (Cohen's d): {round(cohen_d, 4)}. 이는 두 그룹 간의 차이 크기를 나타냅니다.")
                else:
                    interpretation_parts.append(f"• t-test 결과: t-값 = {round(t_stat, 4)}, 자유도 = {int(round(dfree))}, p-값 = {round(p_value, 4)}. p-값이 유의수준 0.05 이상이므로, 두 그룹({col1}, {col2}) 간에는 통계적으로 유의미한 차이가 없습니다.")
                interpretation = "<br>".join(interpretation_parts)

            elif analysis_method == 'anova':
                if df.shape[1] < 3:
                    return 'ANOVA를 수행하려면 세 개 이상의 컬럼이 필요합니다.'
                
                df_long = pd.DataFrame()
                group_stats = {}
                for col in df.columns:
                    temp_df = pd.DataFrame({'value': df[col].dropna(), 'group': col})
                    df_long = pd.concat([df_long, temp_df])
                    
                    group_data = df[col].dropna()
                    group_stats[col] = {
                        'mean': round(group_data.mean(), 4),
                        'std': round(group_data.std(ddof=1), 4),
                        'var': round(group_data.var(ddof=1), 4)
                    }
                
                # ANOVA를 위한 등분산성 검정 (Levene's test)
                # stats.levene는 여러 배열을 인자로 받을 수 있음
                levene_stat, levene_p = stats.levene(*[group for group in all_groups_data if not group.empty])
                homoscedasticity_result = {
                    'statistic': round(levene_stat, 4),
                    'p_value': round(levene_p, 4),
                    'equal_variance': levene_p > 0.05
                }

                try:
                    model = ols('value ~ C(group)', data=df_long).fit()
                    anova_table = sm.stats.anova_lm(model, typ=2) 
                    
                    f_statistic = anova_table['F']['C(group)']
                    p_value = anova_table['PR(>F)']['C(group)']
                    df_between = anova_table['df']['C(group)']
                    df_within = anova_table['df']['Residual']

                    result = {
                        'type': 'anova',
                        'f_statistic': round(f_statistic, 4),
                        'p_value': round(p_value, 4),
                        'df_between': int(df_between),
                        'df_within': int(df_within),
                        'alpha': 0.05,
                        'significant': p_value < 0.05,
                        'group_stats': group_stats,
                        'normality_results': normality_results,
                        'homoscedasticity_result': homoscedasticity_result
                    }
                    
                    # 해석 생성
                    interpretation_parts = []
                    interpretation_parts.append(f"<h3>ANOVA 결과 해석</h3>")
                    
                    # 정규성 해석
                    normality_satisfied = True
                    for col, res in normality_results.items():
                        if not res['normal']:
                            normality_satisfied = False
                            interpretation_parts.append(f"• 그룹 '{col}'의 정규성 검정 (Shapiro-Wilk): 통계량 = {res['statistic']}, p-값 = {res['p_value']}. p-값이 0.05 이하이므로 정규성 가정을 만족하지 않습니다.")
                        else:
                            interpretation_parts.append(f"• 그룹 '{col}'의 정규성 검정 (Shapiro-Wilk): 통계량 = {res['statistic']}, p-값 = {res['p_value']}. p-값이 0.05 초과이므로 정규성 가정을 만족합니다.")
                    
                    if not normality_satisfied:
                        interpretation_parts.append("정규성 가정이 만족되지 않았으므로, ANOVA 결과 해석에 유의해야 합니다. 데이터 변환이나 비모수 검정(예: Kruskal-Wallis H test)을 고려할 수 있습니다.")

                    # 등분산성 해석
                    if homoscedasticity_result['equal_variance']:
                        interpretation_parts.append(f"• 등분산성 검정 (Levene): 통계량 = {homoscedasticity_result['statistic']}, p-값 = {homoscedasticity_result['p_value']}. p-값이 0.05 초과이므로 등분산성 가정을 만족합니다.")
                    else:
                        interpretation_parts.append(f"• 등분산성 검정 (Levene): 통계량 = {homoscedasticity_result['statistic']}, p-값 = {homoscedasticity_result['p_value']}. p-값이 0.05 이하이므로 등분산성 가정을 만족하지 않습니다.")
                        interpretation_parts.append("등분산성 가정이 만족되지 않았으므로, ANOVA 결과 해석에 유의해야 합니다. Welch's ANOVA나 비모수 검정을 고려할 수 있습니다.")

                    # ANOVA 결과 해석
                    if p_value < 0.05:
                        interpretation_parts.append(f"• ANOVA 결과: F-값 = {round(f_statistic, 4)}, 그룹 간 자유도 = {int(df_between)}, 그룹 내 자유도 = {int(df_within)}, p-값 = {round(p_value, 4)}. p-값이 유의수준 0.05보다 작으므로, 그룹 간에 통계적으로 유의미한 차이가 있습니다.")
                        interpretation_parts.append("사후 분석(Post-hoc test)을 통해 어떤 그룹들 사이에 차이가 있는지 추가적으로 확인해야 합니다.")
                    else:
                        interpretation_parts.append(f"• ANOVA 결과: F-값 = {round(f_statistic, 4)}, 그룹 간 자유도 = {int(df_between)}, 그룹 내 자유도 = {int(df_within)}, p-값 = {round(p_value, 4)}. p-값이 유의수준 0.05 이상이므로, 그룹 간에 통계적으로 유의미한 차이가 없습니다.")
                    interpretation = "<br>".join(interpretation_parts)

                except Exception as e:
                    return f"ANOVA 계산 중 오류 발생: {e}. 데이터 형식을 확인해주세요. (예: 모든 그룹에 데이터가 충분한지, 숫자 데이터인지 등)"
            else:
                return '유효하지 않은 분석 방법입니다.'

    except Exception as e:
        return f"파일 처리 중 오류 발생: {e}. 파일 형식을 확인하거나 유효한 데이터를 포함하고 있는지 확인해주세요."
    
    return render_template('index.html', result=result, interpretation=interpretation, 
                            filename=filename, selected_method=selected_method, plot_url=plot_url)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)



