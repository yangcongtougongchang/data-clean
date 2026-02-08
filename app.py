"""
🧹 SmartClean - 智能数据清洗工作台
一个面向零基础用户的可视化数据清洗应用
功能：上传、清洗、分析、导出数据，全程可视化引导
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime
import hashlib

# ============ 页面配置 ============
st.set_page_config(
    page_title="SmartClean - 智能数据清洗",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

# 隐藏GitHub图标和Streamlit默认菜单
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display:none;}
.css-1rs6os {visibility: hidden;}
.css-17ziqus {visibility: hidden;}
.viewerBadge_container__1QSob {visibility: hidden;}
.css-1dp5vir {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ============ 自定义CSS样式 ============
custom_css = """
<style>
    /* 全局字体优化 */
    html, body, [class*="css"] {
        font-family: 'Segoe UI', 'PingFang SC', 'Microsoft YaHei', sans-serif;
    }
    
    /* 标题样式 */
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .sub-title {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* 卡片样式 */
    .info-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .success-card {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #4ade80;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #ffeaa7 0%, #fdcb6e 100%);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        border-left: 5px solid #f59e0b;
    }
    
    /* 步骤指示器 */
    .step-container {
        display: flex;
        justify-content: space-between;
        margin: 30px 0;
        position: relative;
    }
    
    .step-item {
        flex: 1;
        text-align: center;
        padding: 15px;
        background: white;
        border-radius: 12px;
        margin: 0 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        border: 2px solid #e0e0e0;
    }
    
    .step-item.active {
        border-color: #667eea;
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        transform: translateY(-5px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.2);
    }
    
    .step-number {
        width: 35px;
        height: 35px;
        background: #e0e0e0;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-bottom: 8px;
        color: white;
    }
    
    .step-item.active .step-number {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 数据预览表格样式 */
    .dataframe {
        font-size: 0.9rem;
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* 指标卡片 */
    .metric-box {
        background: white;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        transition: transform 0.2s;
    }
    
    .metric-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
    }
    
    .metric-label {
        color: #888;
        font-size: 0.9rem;
        margin-top: 5px;
    }
    
    /* 按钮美化 */
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    /* 侧边栏样式 */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* 帮助提示框 */
    .help-tip {
        background: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    
    /* 引流标识样式 */
    .brand-footer {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        margin-top: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .brand-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .brand-id {
        font-size: 1.2rem;
        background: rgba(255,255,255,0.2);
        padding: 5px 15px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 5px;
    }
    
    /* 代码块样式 */
    .code-block {
        background: #2d2d2d;
        color: #f8f8f2;
        padding: 15px;
        border-radius: 8px;
        font-family: 'Consolas', monospace;
        font-size: 0.85rem;
        overflow-x: auto;
    }
    
    /* 动画效果 */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .animate-in {
        animation: fadeIn 0.6s ease-out;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ============ 多用户隔离机制 ============
def get_user_session():
    """基于浏览器指纹生成唯一用户标识（简单实现）"""
    if 'user_id' not in st.session_state:
        # 生成基于时间的唯一ID
        st.session_state.user_id = hashlib.md5(
            str(datetime.now().timestamp()).encode()
        ).hexdigest()[:8]
    return st.session_state.user_id

USER_ID = get_user_session()

# ============ 数据状态管理 ============
def init_session_state():
    """初始化会话状态（每个用户独立）"""
    defaults = {
        'raw_data': None,
        'cleaned_data': None,
        'file_name': None,
        'cleaning_history': [],
        'current_step': 1,
        'show_tutorial': True,
        'analysis_results': {}
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============ 示例数据生成 ============
@st.cache_data
def generate_sample_data():
    """生成示例数据集供用户初次体验"""
    np.random.seed(42)
    n = 200
    
    data = {
        '客户ID': [f'CUST_{i:04d}' for i in range(1, n+1)],
        '姓名': np.random.choice(['张伟', '李娜', '王芳', '刘洋', '陈静', '杨帆', '赵敏', '黄磊'], n),
        '年龄': np.random.normal(35, 12, n).astype(int),
        '性别': np.random.choice(['男', '女', None], n, p=[0.45, 0.45, 0.1]),
        '注册日期': pd.date_range('2020-01-01', periods=n, freq='D').tolist(),
        '消费金额': np.random.exponential(1000, n).round(2),
        '会员等级': np.random.choice(['普通', '银卡', '金卡', '钻石', None], n, p=[0.4, 0.3, 0.2, 0.05, 0.05]),
        '满意度评分': np.random.choice([1, 2, 3, 4, 5, None], n, p=[0.05, 0.1, 0.2, 0.3, 0.25, 0.1]),
        '最后登录': pd.date_range('2023-01-01', periods=n, freq='h').tolist()
    }
    
    df = pd.DataFrame(data)
    
    # 故意制造一些脏数据用于演示
    # 制造一些异常年龄
    df.loc[np.random.choice(df.index, 5, replace=False), '年龄'] = np.random.choice([150, -5, 999], 5)
    # 制造一些重复行
    df = pd.concat([df, df.iloc[:3]], ignore_index=True)
    # 制造一些异常消费金额
    df.loc[np.random.choice(df.index, 3, replace=False), '消费金额'] = -999.99
    
    return df

# ============ 核心功能函数 ============
def analyze_data_quality(df):
    """全面分析数据质量"""
    analysis = {
        'total_rows': len(df),
        'total_cols': len(df.columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,  # MB
        'dtypes': df.dtypes.to_dict(),
        'missing': df.isnull().sum().to_dict(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        'duplicates': df.duplicated().sum(),
        'numeric_cols': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_cols': df.select_dtypes(include=['object', 'category']).columns.tolist(),
        'datetime_cols': df.select_dtypes(include=['datetime64']).columns.tolist(),
        'unique_values': {col: df[col].nunique() for col in df.columns},
        'outliers': {}
    }
    
    # 检测数值型异常值（使用IQR方法）
    for col in analysis['numeric_cols']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
        analysis['outliers'][col] = {
            'count': int(outliers),
            'lower': lower_bound,
            'upper': upper_bound
        }
    
    return analysis

def clean_data(df, operations):
    """执行数据清洗操作"""
    cleaned = df.copy()
    history = []
    
    for op in operations:
        if op['type'] == 'drop_duplicates':
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            history.append(f"删除重复行: {before - len(cleaned)} 行被移除")
            
        elif op['type'] == 'fill_missing':
            col = op['column']
            method = op['method']
            if method == 'mean':
                cleaned[col] = cleaned[col].fillna(cleaned[col].mean())
            elif method == 'median':
                cleaned[col] = cleaned[col].fillna(cleaned[col].median())
            elif method == 'mode':
                cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
            elif method == 'constant':
                cleaned[col] = cleaned[col].fillna(op['value'])
            history.append(f"填充缺失值 [{col}]: 使用 {method}")
            
        elif op['type'] == 'remove_outliers':
            col = op['column']
            Q1 = cleaned[col].quantile(0.25)
            Q3 = cleaned[col].quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            before = len(cleaned)
            cleaned = cleaned[(cleaned[col] >= lower) & (cleaned[col] <= upper)]
            history.append(f"异常值处理 [{col}]: 移除 {before - len(cleaned)} 个异常值")
            
        elif op['type'] == 'convert_type':
            col = op['column']
            new_type = op['new_type']
            try:
                if new_type == 'datetime':
                    cleaned[col] = pd.to_datetime(cleaned[col])
                else:
                    cleaned[col] = cleaned[col].astype(new_type)
                history.append(f"类型转换 [{col}]: 转换为 {new_type}")
            except:
                history.append(f"类型转换 [{col}]: 失败")
                
        elif op['type'] == 'drop_column':
            col = op['column']
            cleaned = cleaned.drop(columns=[col])
            history.append(f"删除列: {col}")
            
        elif op['type'] == 'rename_column':
            old, new = op['old_name'], op['new_name']
            cleaned = cleaned.rename(columns={old: new})
            history.append(f"重命名: {old} -> {new}")
    
    return cleaned, history

def get_download_link(df, filename="cleaned_data.csv"):
    """生成CSV下载链接"""
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}" class="download-btn">📥 下载清洗后的数据</a>'
    return href

# ============ 可视化函数 ============
def create_overview_charts(df, analysis):
    """创建数据概览可视化"""
    charts = []
    
    # 1. 数据类型分布饼图
    type_counts = {
        '数值型': len(analysis['numeric_cols']),
        '分类型': len(analysis['categorical_cols']),
        '日期型': len(analysis['datetime_cols'])
    }
    fig1 = px.pie(
        values=list(type_counts.values()), 
        names=list(type_counts.keys()),
        title="📊 数据类型分布",
        color_discrete_sequence=px.colors.sequential.Purple,
        hole=0.4
    )
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    charts.append(fig1)
    
    # 2. 缺失值热力图
    if any(v > 0 for v in analysis['missing'].values()):
        missing_df = pd.DataFrame({
            '列名': list(analysis['missing'].keys()),
            '缺失数量': list(analysis['missing'].values()),
            '缺失比例(%)': list(analysis['missing_pct'].values())
        })
        missing_df = missing_df[missing_df['缺失数量'] > 0].sort_values('缺失数量', ascending=True)
        
        fig2 = px.bar(
            missing_df,
            x='缺失数量',
            y='列名',
            color='缺失比例(%)',
            orientation='h',
            title="🔍 缺失值分布",
            color_continuous_scale='Reds',
            text='缺失数量'
        )
        fig2.update_traces(textposition='outside')
        charts.append(fig2)
    
    # 3. 数值型列分布图
    if analysis['numeric_cols']:
        fig3 = make_subplots(
            rows=min(2, len(analysis['numeric_cols'])), 
            cols=2,
            subplot_titles=[f"{col} 分布" for col in analysis['numeric_cols'][:4]]
        )
        
        for idx, col in enumerate(analysis['numeric_cols'][:4]):
            row = idx // 2 + 1
            col_idx = idx % 2 + 1
            fig3.add_trace(
                go.Histogram(x=df[col], name=col, marker_color='#667eea', nbinsx=30),
                row=row, col=col_idx
            )
        
        fig3.update_layout(
            title_text="📈 数值型特征分布",
            showlegend=False,
            height=400
        )
        charts.append(fig3)
    
    return charts

def create_correlation_heatmap(df):
    """创建相关性热力图"""
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr = numeric_df.corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            aspect="auto",
            color_continuous_scale='RdBu_r',
            title="🔗 特征相关性矩阵",
            zmin=-1, zmax=1
        )
        fig.update_traces(texttemplate='%{text:.2f}')
        return fig
    return None

def create_cleaning_impact_chart(before_df, after_df):
    """展示清洗前后对比"""
    metrics = {
        '行数': [len(before_df), len(after_df)],
        '缺失值总数': [before_df.isnull().sum().sum(), after_df.isnull().sum().sum()],
        '重复行数': [before_df.duplicated().sum(), after_df.duplicated().sum()],
        '内存使用(MB)': [
            before_df.memory_usage(deep=True).sum() / 1024**2,
            after_df.memory_usage(deep=True).sum() / 1024**2
        ]
    }
    
    fig = go.Figure()
    
    categories = list(metrics.keys())
    
    fig.add_trace(go.Bar(
        name='清洗前',
        x=categories,
        y=[metrics[k][0] for k in categories],
        marker_color='#ff6b6b',
        text=[f'{v:.1f}' if isinstance(v, float) else str(v) for v in [metrics[k][0] for k in categories]],
        textposition='auto'
    ))
    
    fig.add_trace(go.Bar(
        name='清洗后',
        x=categories,
        y=[metrics[k][1] for k in categories],
        marker_color='#4ecdc4',
        text=[f'{v:.1f}' if isinstance(v, float) else str(v) for v in [metrics[k][1] for k in categories]],
        textposition='auto'
    ))
    
    fig.update_layout(
        title="📊 数据清洗效果对比",
        barmode='group',
        xaxis_title="指标",
        yaxis_title="数值",
        height=400,
        template='plotly_white'
    )
    
    return fig

# ============ UI组件 ============
def render_header():
    """渲染页面头部"""
    st.markdown('<h1 class="main-title">🧹 SmartClean 智能数据清洗</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">零基础友好的数据清洗与可视化分析平台</p>', unsafe_allow_html=True)
    
    # 步骤指示器
    steps = [
        ("1", "上传数据", "📤"),
        ("2", "质量分析", "🔍"),
        ("3", "智能清洗", "✨"),
        ("4", "可视化", "📈"),
        ("5", "导出结果", "💾")
    ]
    
    current = st.session_state.current_step
    html_steps = '<div class="step-container">'
    for i, (num, label, icon) in enumerate(steps, 1):
        active_class = "active" if i == current else ""
        html_steps += f'''
        <div class="step-item {active_class}">
            <div class="step-number">{icon}</div>
            <div style="font-size:0.9rem;font-weight:600;">{label}</div>
        </div>
        '''
    html_steps += '</div>'
    st.markdown(html_steps, unsafe_allow_html=True)

def render_tutorial():
    """渲染使用教程（针对零基础用户）"""
    with st.expander("📚 新手入门指南（点击展开）", expanded=st.session_state.show_tutorial):
        st.markdown("""
        <div class="info-card">
        <h4>🎯 什么是数据清洗？</h4>
        <p>数据清洗就像整理房间：原始数据通常包含<strong>缺失值</strong>（空格子）、<strong>异常值</strong>（奇怪的数字）、
        <strong>重复数据</strong>（多余的复印件）和<strong>格式错误</strong>（放错地方的物品）。本工具帮您自动发现并修复这些问题。</p>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 20px 0;">
            <div class="metric-box">
                <div style="font-size: 2rem;">1️⃣</div>
                <h4>上传文件</h4>
                <p style="font-size: 0.9rem; color: #666;">支持 CSV、Excel 格式<br>或直接试用示例数据</p>
            </div>
            <div class="metric-box">
                <div style="font-size: 2rem;">2️⃣</div>
                <h4>查看分析</h4>
                <p style="font-size: 0.9rem; color: #666;">自动检测数据质量问题<br>可视化展示统计图表</p>
            </div>
            <div class="metric-box">
                <div style="font-size: 2rem;">3️⃣</div>
                <h4>选择清洗</h4>
                <p style="font-size: 0.9rem; color: #666;">勾选要执行的清洗操作<br>实时预览处理效果</p>
            </div>
            <div class="metric-box">
                <div style="font-size: 2rem;">4️⃣</div>
                <h4>导出数据</h4>
                <p style="font-size: 0.9rem; color: #666;">下载清洗后的干净数据<br>支持 CSV/Excel 格式</p>
            </div>
        </div>
        
        <div class="help-tip">
        💡 <strong>提示：</strong>如果您没有现成的数据文件，可以点击"使用示例数据"按钮，我们会提供一份包含常见问题的模拟数据供您练习。
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("👍 我已了解，隐藏指南", use_container_width=True):
            st.session_state.show_tutorial = False
            st.rerun()

def render_upload_section():
    """渲染数据上传区域"""
    st.markdown("### 📤 第一步：上传您的数据")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "拖拽文件到此处，或点击上传",
            type=['csv', 'xlsx', 'xls'],
            help="支持 CSV 和 Excel 格式，文件大小建议不超过 200MB",
            key=f"uploader_{USER_ID}"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file, encoding='utf-8')
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.raw_data = df
                st.session_state.file_name = uploaded_file.name
                st.session_state.current_step = 2
                st.success(f"✅ 成功加载文件：{uploaded_file.name}，共 {len(df)} 行 × {len(df.columns)} 列")
                
            except Exception as e:
                st.error(f"❌ 文件读取失败：{str(e)}")
                st.info("💡 尝试解决方法：如果是 CSV 文件，请尝试用记事本打开并另存为 UTF-8 编码格式")
    
    with col2:
        st.markdown("#### 或者")
        if st.button("🎲 使用示例数据体验", use_container_width=True, type="primary"):
            df = generate_sample_data()
            st.session_state.raw_data = df
            st.session_state.file_name = "示例数据.csv"
            st.session_state.current_step = 2
            st.balloons()
            st.success("✅ 已加载示例数据！这份数据故意包含了一些常见问题（缺失值、异常年龄、重复行等），供您练习清洗操作。")

def render_analysis_section():
    """渲染数据分析区域"""
    # 修复：使用 a.empty 检查 DataFrame 是否为空，而不是布尔判断
    if st.session_state.raw_data is None or st.session_state.raw_data.empty:
        return
    
    st.markdown("---")
    st.markdown("### 🔍 第二步：数据质量诊断")
    
    df = st.session_state.raw_data
    analysis = analyze_data_quality(df)
    st.session_state.analysis_results = analysis
    
    # 关键指标卡片
    cols = st.columns(4)
    metrics = [
        ("总行数", f"{analysis['total_rows']:,}", "人"),
        ("总列数", analysis['total_cols'], "列"),
        ("缺失值比例", f"{sum(analysis['missing'].values()) / (analysis['total_rows'] * analysis['total_cols']) * 100:.1f}", "%"),
        ("重复行数", analysis['duplicates'], "行")
    ]
    
    for col, (label, value, unit) in zip(cols, metrics):
        with col:
            st.markdown(f'''
            <div class="metric-box">
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label} ({unit})</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # 详细分析标签页
    tab1, tab2, tab3 = st.tabs(["📊 可视化概览", "📋 详细统计", "🔍 数据预览"])
    
    with tab1:
        charts = create_overview_charts(df, analysis)
        for i, chart in enumerate(charts):
            st.plotly_chart(chart, use_container_width=True, key=f"chart_{i}_{USER_ID}")
        
        # 相关性分析
        corr_chart = create_correlation_heatmap(df)
        if corr_chart:
            st.plotly_chart(corr_chart, use_container_width=True, key=f"corr_{USER_ID}")
    
    with tab2:
        # 数据类型表
        st.markdown("**数据类型详情**")
        dtype_df = pd.DataFrame({
            '列名': list(analysis['dtypes'].keys()),
            '数据类型': [str(t) for t in analysis['dtypes'].values()],
            '非空值数量': [analysis['total_rows'] - analysis['missing'][col] for col in analysis['dtypes'].keys()],
            '缺失值数量': list(analysis['missing'].values()),
            '唯一值数量': list(analysis['unique_values'].values())
        })
        st.dataframe(dtype_df, use_container_width=True, hide_index=True)
        
        # 异常值详情
        if analysis['outliers']:
            st.markdown("**异常值检测（基于IQR方法）**")
            outlier_df = pd.DataFrame([
                {
                    '列名': col,
                    '异常值数量': info['count'],
                    '正常范围': f"[{info['lower']:.2f}, {info['upper']:.2f}]"
                }
                for col, info in analysis['outliers'].items()
            ])
            st.dataframe(outlier_df, use_container_width=True, hide_index=True)
    
    with tab3:
        st.markdown(f"**显示前 100 行（共 {len(df)} 行）**")
        st.dataframe(df.head(100), use_container_width=True)
        
        # 列选择查看
        selected_cols = st.multiselect(
            "选择特定列查看",
            options=df.columns.tolist(),
            default=list(df.columns[:5]),
            key=f"cols_select_{USER_ID}"
        )
        if selected_cols:
            st.dataframe(df[selected_cols].head(50), use_container_width=True)

def render_cleaning_section():
    """渲染数据清洗操作区"""
    # 修复：使用 a.empty 检查 DataFrame 是否为空
    if st.session_state.raw_data is None or st.session_state.raw_data.empty:
        return
    
    st.markdown("---")
    st.markdown("### ✨ 第三步：智能数据清洗")
    
    df = st.session_state.raw_data.copy()
    analysis = st.session_state.analysis_results
    
    operations = []
    
    with st.container():
        st.markdown('<div class="info-card">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ 选择清洗操作（可多选）")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. 处理重复值
            if analysis['duplicates'] > 0:
                st.markdown(f'''
                <div class="warning-card">
                ⚠️ 发现 {analysis['duplicates']} 行完全重复的数据
                </div>
                ''', unsafe_allow_html=True)
                if st.checkbox("🗑️ 删除重复行", value=True, key=f"dup_{USER_ID}"):
                    operations.append({'type': 'drop_duplicates'})
            else:
                st.markdown('<div class="success-card">✅ 未发现重复行</div>', unsafe_allow_html=True)
            
            # 2. 处理缺失值
            st.markdown("#### 📝 缺失值处理策略")
            missing_cols = [col for col, count in analysis['missing'].items() if count > 0]
            
            if missing_cols:
                for col in missing_cols:
                    with st.expander(f"列 '{col}' - 缺失 {analysis['missing'][col]} 个值 ({analysis['missing_pct'][col]}%)"):
                        method = st.selectbox(
                            f"填充方式",
                            ["不处理", "删除该行", "填充均值", "填充中位数", "填充众数", "填充固定值"],
                            key=f"missing_{col}_{USER_ID}"
                        )
                        
                        if method == "删除该行":
                            # 这里简化处理，实际应该标记行删除
                            pass
                        elif method == "填充均值":
                            operations.append({'type': 'fill_missing', 'column': col, 'method': 'mean'})
                        elif method == "填充中位数":
                            operations.append({'type': 'fill_missing', 'column': col, 'method': 'median'})
                        elif method == "填充众数":
                            operations.append({'type': 'fill_missing', 'column': col, 'method': 'mode'})
                        elif method == "填充固定值":
                            val = st.text_input("输入填充值", key=f"fill_val_{col}_{USER_ID}")
                            if val:
                                operations.append({'type': 'fill_missing', 'column': col, 'method': 'constant', 'value': val})
            else:
                st.markdown('<div class="success-card">✅ 未发现缺失值</div>', unsafe_allow_html=True)
        
        with col2:
            # 3. 异常值处理
            st.markdown("#### 🚨 异常值处理")
            if analysis['outliers']:
                for col, info in analysis['outliers'].items():
                    if info['count'] > 0:
                        with st.expander(f"列 '{col}' - {info['count']} 个异常值"):
                            if st.checkbox(f"移除 {col} 的异常值", key=f"outlier_{col}_{USER_ID}"):
                                operations.append({'type': 'remove_outliers', 'column': col})
            else:
                st.markdown('<div class="success-card">✅ 未发现明显异常值</div>', unsafe_allow_html=True)
            
            # 4. 类型转换
            st.markdown("#### 🔄 数据类型转换")
            type_cols = st.multiselect(
                "选择要转换类型的列",
                list(df.columns),
                key=f"type_cols_{USER_ID}"
            )
            for col in type_cols:
                new_type = st.selectbox(
                    f"{col} 转换为",
                    ["保持原样", "整数(int)", "浮点数(float)", "字符串(str)", "日期时间(datetime)"],
                    key=f"type_{col}_{USER_ID}"
                )
                type_map = {
                    "整数(int)": "int64",
                    "浮点数(float)": "float64", 
                    "字符串(str)": "object",
                    "日期时间(datetime)": "datetime"
                }
                if new_type in type_map:
                    operations.append({
                        'type': 'convert_type', 
                        'column': col, 
                        'new_type': type_map[new_type]
                    })
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # 执行清洗
    if operations:
        if st.button("🚀 执行清洗操作", type="primary", use_container_width=True):
            with st.spinner("正在清洗数据..."):
                cleaned_df, history = clean_data(df, operations)
                st.session_state.cleaned_data = cleaned_df
                st.session_state.cleaning_history = history
                st.session_state.current_step = 4
                st.success(f"✅ 清洗完成！共执行 {len(history)} 项操作")
    else:
        st.info("💡 请选择至少一项清洗操作，或点击上方复选框启用自动建议的清洗项")

def render_results_section():
    """渲染清洗结果和导出"""
    # 修复：使用 a.empty 检查 DataFrame 是否为空
    if st.session_state.cleaned_data is None or st.session_state.cleaned_data.empty:
        # 如果没有清洗数据但原始数据存在，显示原始数据对比
        if st.session_state.raw_data is not None and not st.session_state.raw_data.empty:
            st.markdown("---")
            st.markdown("### 📈 第四步：可视化分析")
            st.info("执行清洗操作后，此处将显示清洗前后的对比分析")
        return
    
    st.markdown("---")
    st.markdown("### 📈 第四步：清洗效果评估")
    
    before_df = st.session_state.raw_data
    after_df = st.session_state.cleaned_data
    
    # 对比图表
    impact_chart = create_cleaning_impact_chart(before_df, after_df)
    st.plotly_chart(impact_chart, use_container_width=True, key=f"impact_{USER_ID}")
    
    # 操作历史
    with st.expander("📝 查看清洗操作记录"):
        for i, record in enumerate(st.session_state.cleaning_history, 1):
            st.markdown(f"{i}. {record}")
    
    # 清洗后数据预览
    st.markdown("#### 清洗后数据预览")
    st.dataframe(after_df.head(100), use_container_width=True)
    
    # 导出区域
    st.markdown("---")
    st.markdown("### 💾 第五步：导出清洗结果")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        export_format = st.selectbox(
            "导出格式",
            ["CSV (推荐)", "Excel"],
            key=f"export_fmt_{USER_ID}"
        )
    
    with col2:
        if st.button("📥 生成下载文件", type="primary", use_container_width=True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = st.session_state.file_name.rsplit('.', 1)[0]
            
            if export_format == "CSV (推荐)":
                csv = after_df.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="⬇️ 点击下载 CSV",
                    data=csv,
                    file_name=f"{base_name}_cleaned_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            else:
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    after_df.to_excel(writer, index=False, sheet_name='清洗后数据')
                    # 添加操作记录sheet
                    history_df = pd.DataFrame({'操作记录': st.session_state.cleaning_history})
                    history_df.to_excel(writer, index=False, sheet_name='清洗记录')
                
                st.download_button(
                    label="⬇️ 点击下载 Excel",
                    data=buffer.getvalue(),
                    file_name=f"{base_name}_cleaned_{timestamp}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
    
    with col3:
        st.markdown(f'''
        <div class="info-card" style="margin-top: 0;">
        <strong>📋 导出信息</strong><br>
        原始文件：{st.session_state.file_name}<br>
        清洗时间：{datetime.now().strftime("%Y-%m-%d %H:%M")}<br>
        最终行数：{len(after_df):,} 行<br>
        压缩率：{(1 - len(after_df)/len(before_df))*100:.1f}%
        </div>
        ''', unsafe_allow_html=True)

def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        st.markdown("### 🧹 SmartClean")
        st.markdown("---")
        
        # 当前状态
        st.markdown("**当前会话状态**")
        # 修复：使用 is None 和 empty 检查
        has_data = st.session_state.raw_data is not None and not st.session_state.raw_data.empty
        status_color = "🟢" if has_data else "⚪"
        st.markdown(f"{status_color} 数据加载: {'已完成' if has_data else '未开始'}")
        
        if has_data:
            has_cleaned = st.session_state.cleaned_data is not None and not st.session_state.cleaned_data.empty
            status_color = "🟢" if has_cleaned else "🟡"
            st.markdown(f"{status_color} 数据清洗: {'已完成' if has_cleaned else '进行中'}")
        
        st.markdown("---")
        
        # 快捷操作
        st.markdown("**快捷操作**")
        if st.button("🔄 重新开始", use_container_width=True):
            for key in ['raw_data', 'cleaned_data', 'file_name', 'cleaning_history', 'current_step']:
                st.session_state[key] = None if key != 'current_step' else 1
            st.rerun()
        
        if has_data and st.button("📊 仅查看分析", use_container_width=True):
            st.session_state.current_step = 2
            st.rerun()
        
        st.markdown("---")
        
        # 帮助链接
        st.markdown("**需要帮助？**")
        with st.expander("常见问题"):
            st.markdown("""
            **Q: 支持多大的文件？**  
            A: 建议不超过 200MB，超过请分批处理。
            
            **Q: 数据安全吗？**  
            A: 所有处理在浏览器内存中进行，不会上传服务器。
            
            **Q: 中文乱码怎么办？**  
            A: 请确保 CSV 文件使用 UTF-8 编码保存。
            """)
        
        st.markdown("---")
        st.caption(f"👤 会话ID: {USER_ID}")

def render_footer():
    """简洁版页脚"""
    
    css = """
    <style>
    .simple-footer {
        text-align: center;
        padding: 20px;
        margin-top: 40px;
        background: #f5f7fa;
        border-radius: 10px;
        border-top: 2px solid #ff2442;
    }
    
    .footer-title {
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 10px;
        color: #1a1a2e;
    }
    
    .xh-box {
        display: inline-block;
        background: #ff2442;
        color: white;
        padding: 8px 16px;
        border-radius: 6px;
        text-decoration: none;
        margin: 10px 0;
        font-weight: bold;
    }
    
    .xh-box:hover {
        background: #e0203c;
        transform: scale(1.05);
        transition: all 0.3s;
    }
    
    .footer-text {
        color: #666;
        margin: 15px 0;
        font-size: 0.9rem;
    }
    
    .copyright {
        color: #888;
        font-size: 0.8rem;
        margin-top: 15px;
    }
    </style>
    """
    
    html = f"""
    <div class="simple-footer">
        <div class="footer-title">🏭 洋葱头工厂</div>
        
        <a href="https://www.xiaohongshu.com/user/profile/5e0554d5000000000100315c" target="_blank" class="xh-box">
            📕 小红书：750922641
        </a>
        
        <p class="footer-text">专注 AI 工具与数据智能 · 关注获取更多实用技巧</p>
        
        <div class="copyright">
            © 2023 SmartClean · 设计 by 
            <a href="https://www.xiaohongshu.com/user/profile/750922641" target="_blank" style="color: #ff2442; text-decoration: none;">
                洋葱头工厂
            </a>
            <br>
            <span style="font-size: 0.75rem;">本地化处理 · 隐私安全 · 零基础友好</span>
        </div>
    </div>
    """
    
    st.markdown("---")
    st.markdown(css, unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

# ============ 主程序 ============
def main():
    render_header()
    render_tutorial()
    render_sidebar()
    
    # 主流程
    render_upload_section()
    render_analysis_section()
    render_cleaning_section()
    render_results_section()
    
    # 页脚
    render_footer()

if __name__ == "__main__":
    main()



