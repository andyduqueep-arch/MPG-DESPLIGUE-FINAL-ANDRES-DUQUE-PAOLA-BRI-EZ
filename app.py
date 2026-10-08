"""
App Streamlit: predicción del consumo de combustible (mpg) con el modelo de bagging optimizado.
Archivo único: no depende de otros .py. Debe estar junto a:
  one_hot_columns.joblib, min_max_scaler.joblib, bagging_optimizado.joblib y requirements.txt
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# =============================================================== preprocesamiento y predicción
TARGET = 'mpg'
CATEGORICAS = {'origin': ['USA', 'Europe', 'Japan']}
ARCHIVOS = ('one_hot_columns.joblib', 'min_max_scaler.joblib', 'bagging_optimizado.joblib')

# Rango y valor por defecto de cada variable (Dataset_limpio.xlsx y reglas de calidad del punto 2.3)
RANGOS = {
    'cylinders':    dict(min=3,      max=8,      default=4,      step=1,    label='Cilindros'),
    'displacement': dict(min=50.0,   max=500.0,  default=148.8,  step=1.0,  label='Cilindrada (pulg³)'),
    'horsepower':   dict(min=40.0,   max=500.0,  default=102.8,  step=1.0,  label='Potencia (HP)'),
    'weight':       dict(min=1500.0, max=6000.0, default=2680.0, step=10.0, label='Peso (lb)'),
    'acceleration': dict(min=5.0,    max=25.0,   default=15.6,   step=0.1,  label='Aceleración 0-60 mph (s)'),
    'model_year':   dict(min=1970,   max=2026,   default=2010,   step=1,    label='Año del modelo'),
    'vehicle_age':  dict(min=0,      max=56,     default=16,     step=1,    label='Antigüedad (años)'),
}


def buscar_archivo(carpeta, nombre):
    """Encuentra el archivo aunque el navegador lo haya renombrado (p. ej. 'min_max_scaler (1).joblib')."""
    exacto = carpeta / nombre
    if exacto.exists():
        return exacto
    base = nombre.rsplit('.', 1)[0]
    candidatos = sorted(carpeta.glob(base + '*.joblib'))
    if candidatos:
        return candidatos[0]
    raise FileNotFoundError(2, 'No encontrado', nombre)


def cargar_artefactos(carpeta='.'):
    """Carga los 3 archivos .joblib: one-hot, scaler y modelo."""
    c = Path(carpeta)
    return tuple(joblib.load(buscar_archivo(c, f)) for f in ARCHIVOS)


def nombres(obj):
    """Columnas con las que se ajustó un objeto de sklearn (None si se ajustó con arrays)."""
    n = getattr(obj, 'feature_names_in_', None)
    return list(n) if n is not None else None


def es_encoder(obj):
    """True si one_hot_columns.joblib guarda un OneHotEncoder (y no una lista de columnas)."""
    return hasattr(obj, 'transform') and hasattr(obj, 'categories_')


def columnas_modelo(one_hot, modelo):
    """Orden final de columnas que espera el modelo."""
    if nombres(modelo):
        return nombres(modelo)
    return None if es_encoder(one_hot) else list(one_hot)


def variables_de_entrada(one_hot, scaler, modelo):
    """Deduce de los artefactos qué variables originales hay que pedirle al usuario."""
    cols = columnas_modelo(one_hot, modelo) or []
    if es_encoder(one_hot):
        cats = list(nombres(one_hot) or CATEGORICAS)
        base = nombres(scaler) or cols
    else:
        cats = [k for k in CATEGORICAS if any(c.startswith(k + '_') for c in cols)]
        base = cols
    nums = [c for c in base if c != TARGET and not any(c.startswith(k + '_') for k in cats)]
    return nums, cats


def opciones_categoricas(var, one_hot):
    if es_encoder(one_hot):
        j = (nombres(one_hot) or list(CATEGORICAS)).index(var)
        return [str(x) for x in one_hot.categories_[j]]
    return CATEGORICAS.get(var, [])


def aplicar_one_hot(df, one_hot):
    """Paso 1: One-Hot igual que en el entrenamiento (mismas columnas y mismo orden)."""
    if es_encoder(one_hot):
        cat_cols = nombres(one_hot) or list(CATEGORICAS)
        arr = one_hot.transform(df[cat_cols])
        arr = arr.toarray() if hasattr(arr, 'toarray') else arr
        dummies = pd.DataFrame(arr, columns=one_hot.get_feature_names_out(cat_cols), index=df.index)
        return pd.concat([df.drop(columns=cat_cols), dummies], axis=1)
    cat_cols = [c for c in CATEGORICAS if c in df.columns]
    df = pd.get_dummies(df, columns=cat_cols, dtype=int)
    return df.reindex(columns=list(one_hot), fill_value=0)   # agrega las dummies faltantes con 0


def aplicar_scaler(df, scaler):
    """Paso 2: MinMaxScaler sólo sobre las columnas con las que se ajustó."""
    df = df.copy()
    cols = nombres(scaler)
    if cols is None:  # ajustado con array sin nombres
        n = scaler.n_features_in_
        cols = list(df.columns) if n == df.shape[1] else [c for c in df.columns if c in RANGOS][:n]
    tmp = df.reindex(columns=cols, fill_value=0).astype(float)
    esc = pd.DataFrame(scaler.transform(tmp.values), columns=cols, index=df.index)
    for c in cols:
        if c in df.columns:
            df[c] = esc[c]
    return df


def preparar(entrada, one_hot, scaler, modelo):
    """Entrada original (dict o DataFrame) -> matriz lista para el modelo."""
    df = pd.DataFrame([entrada]) if isinstance(entrada, dict) else entrada.copy()
    df = aplicar_scaler(aplicar_one_hot(df, one_hot), scaler)
    orden = columnas_modelo(one_hot, modelo) or list(df.columns)
    return df.reindex(columns=orden, fill_value=0)


def desescalar_objetivo(pred, scaler):
    """Si el scaler incluyó mpg, la predicción sale entre 0 y 1: se devuelve a mpg."""
    cols = nombres(scaler)
    if cols and TARGET in cols:
        i = cols.index(TARGET)
        return np.asarray(pred) * scaler.data_range_[i] + scaler.data_min_[i]
    return np.asarray(pred)


def predecir(entrada, one_hot, scaler, modelo):
    """Devuelve las predicciones de mpg (array) para una o varias filas."""
    X = preparar(entrada, one_hot, scaler, modelo)
    pred = modelo.predict(X if nombres(modelo) else X.values)
    return desescalar_objetivo(pred, scaler)


def predicciones_individuales(modelo, X, scaler=None):
    """Predicción de cada estimador del ensamble para una fila (rango de incertidumbre)."""
    try:
        Xv = X.values
        feats = getattr(modelo, 'estimators_features_', None)
        if feats is not None:  # BaggingRegressor
            p = [e.predict(Xv[:, f])[0] for e, f in zip(modelo.estimators_, feats)]
        else:                  # RandomForest / ExtraTrees
            p = [e.predict(Xv)[0] for e in modelo.estimators_]
        return desescalar_objetivo(p, scaler) if scaler is not None else np.array(p)
    except Exception:
        return None


# =============================================================== interfaz
import io

st.set_page_config(page_title='Predicción de consumo (mpg)', page_icon='⛽', layout='centered')
try:  # color principal minimalista (sin necesitar .streamlit/config.toml)
    st._config.set_option('theme.primaryColor', '#0F766E')
    st._config.set_option('theme.base', 'light')
except Exception:
    pass

# ---------- estilo minimalista ----------
st.markdown("""
<style>
:root{
  --ink:#1E293B; --muted:#64748B; --line:#E2E8F0; --bg:#F8FAFC; --card:#FFFFFF;
  --accent:#0F766E; --accent-soft:#E6F4F1;
  --bajo:#DC6B5A; --medio:#E0A43A; --alto:#2F9E73;
}
.stApp{background:var(--bg);}
.block-container{padding-top:2.2rem; max-width:860px;}
.stMarkdown, label {color:var(--ink);}
h1,h2,h3{color:var(--ink); letter-spacing:-0.01em;}

.hero{background:var(--card); border:1px solid var(--line); border-radius:16px; padding:26px 28px; margin-bottom:18px;}
.hero h1{font-size:1.9rem; margin:0 0 4px 0; font-weight:700;}
.hero .autores{color:var(--accent); font-weight:600; font-size:1.02rem; margin:0 0 10px 0;}
.hero .desc{color:var(--muted); font-size:.92rem; margin:0;}

.stTabs [data-baseweb="tab-list"]{gap:6px; border-bottom:1px solid var(--line);}
.stTabs [data-baseweb="tab"]{background:transparent; border-radius:10px 10px 0 0; padding:10px 18px; color:var(--muted); font-weight:600;}
.stTabs [aria-selected="true"]{color:var(--accent)!important; background:var(--accent-soft);}

div[data-testid="stForm"], div[data-testid="stFileUploader"] section{background:var(--card); border:1px solid var(--line); border-radius:14px;}
div[data-testid="stForm"]{padding:20px 22px;}

.stButton>button, .stFormSubmitButton>button, .stDownloadButton>button{
  border-radius:10px; font-weight:600; border:1px solid var(--accent);}
.stButton>button[kind="primary"], .stFormSubmitButton>button{
  background:var(--accent)!important; border-color:var(--accent)!important; color:#fff!important;}
.stButton>button[kind="primary"] p, .stFormSubmitButton>button p{color:#fff!important;}
.stButton>button[kind="primary"]:hover, .stFormSubmitButton>button:hover{
  background:#0B5F58!important; border-color:#0B5F58!important;}
.stButton>button:disabled{background:#CBD5E1!important; border-color:#CBD5E1!important;}
[data-baseweb="tab-highlight"]{background-color:var(--accent)!important;}
[data-baseweb="tag"]{background-color:var(--accent-soft)!important;}
[data-baseweb="tag"] span{color:var(--accent)!important;}
[data-baseweb="tag"] svg{fill:var(--accent)!important;}
.stDownloadButton>button{background:var(--card); color:var(--accent);}

.kpis{display:grid; grid-template-columns:repeat(3,1fr); gap:12px; margin:8px 0 14px;}
.kpi{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 18px;}
.kpi .lbl{color:var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.04em;}
.kpi .val{font-size:1.7rem; font-weight:700; color:var(--ink); line-height:1.25;}
.kpi .sub{color:var(--muted); font-size:.82rem;}

.medidor{background:var(--card); border:1px solid var(--line); border-radius:14px; padding:16px 18px; margin-bottom:12px;}
.barra{position:relative; height:12px; border-radius:99px;
  background:linear-gradient(90deg,var(--bajo) 0%,var(--medio) 45%,var(--alto) 100%); margin:12px 0 6px;}
.marca{position:absolute; top:-5px; width:4px; height:22px; background:var(--ink); border-radius:2px;}
.escala{display:flex; justify-content:space-between; color:var(--muted); font-size:.78rem;}
.chip{display:inline-block; padding:3px 12px; border-radius:99px; font-weight:600; font-size:.85rem; color:#fff;}

.footer{color:var(--muted); font-size:.8rem; text-align:center; margin-top:28px;}
@media (max-width:640px){.kpis{grid-template-columns:1fr;} .hero h1{font-size:1.5rem;}}
</style>
""", unsafe_allow_html=True)

# ---------- encabezado ----------
st.markdown("""
<div class="hero">
  <h1>⛽ Predicción del consumo de combustible</h1>
  <p class="autores">Por: Andres Duque &amp; Paola Briñez</p>
  <p class="desc">Modelo de bagging optimizado · entrada → One-Hot → MinMaxScaler → predicción de mpg</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def artefactos():
    return cargar_artefactos(Path(__file__).parent)   # busca los .joblib junto a app.py


try:
    one_hot, scaler, modelo = artefactos()
except FileNotFoundError as e:
    st.error(f'No se encontró {e.filename}. Copia los 3 archivos .joblib en la misma carpeta que app.py.')
    st.stop()

nums, cats = variables_de_entrada(one_hot, scaler, modelo)
COLOR_NIVEL = {'bajo': '#DC6B5A', 'medio': '#E0A43A', 'alto': '#2F9E73'}


def nivel_eficiencia(mpg):
    return 'alto' if mpg > 35 else 'medio' if mpg >= 25 else 'bajo'


def tarjetas(mpg):
    st.markdown(f"""
    <div class="kpis">
      <div class="kpi"><div class="lbl">Rendimiento</div><div class="val">{mpg:.1f} mpg</div><div class="sub">millas por galón</div></div>
      <div class="kpi"><div class="lbl">Consumo</div><div class="val">{235.215 / mpg:.1f}</div><div class="sub">litros por 100 km</div></div>
      <div class="kpi"><div class="lbl">vs. promedio</div><div class="val">{mpg - 29.9:+.1f}</div><div class="sub">mpg frente a 29,9 mpg</div></div>
    </div>""", unsafe_allow_html=True)


def medidor(mpg):
    nivel = nivel_eficiencia(mpg)
    pos = min(max((mpg - 9) / (48 - 9), 0), 1) * 100   # rango observado de mpg: 9 a 48
    st.markdown(f"""
    <div class="medidor">
      Eficiencia: <span class="chip" style="background:{COLOR_NIVEL[nivel]}">{nivel.upper()}</span>
      <div class="barra"><div class="marca" style="left:calc({pos:.1f}% - 2px)"></div></div>
      <div class="escala"><span>9 mpg</span><span>bajo &lt; 25</span><span>25 – 35 medio</span><span>alto &gt; 35</span><span>48 mpg</span></div>
    </div>""", unsafe_allow_html=True)


tab_uno, tab_excel = st.tabs(['🚗  Un vehículo', '📄  Predicción desde Excel'])

# =============================================================== pestaña 1: un vehículo
with tab_uno:
    with st.form('vehiculo'):
        st.markdown('#### Características del vehículo')
        entrada = {}
        c1, c2 = st.columns(2)
        for i, var in enumerate(nums):
            r = RANGOS.get(var, dict(min=None, max=None, default=0.0, step=None, label=var))
            entero = isinstance(r['default'], int)
            entrada[var] = (c1 if i % 2 == 0 else c2).number_input(
                r['label'], min_value=r['min'], max_value=r['max'], value=r['default'], step=r['step'],
                format='%d' if entero else '%.1f')
        for var in cats:
            entrada[var] = st.segmented_control('Origen' if var == 'origin' else var,
                                                opciones_categoricas(var, one_hot),
                                                default=opciones_categoricas(var, one_hot)[0]) \
                if hasattr(st, 'segmented_control') else \
                st.selectbox('Origen' if var == 'origin' else var, opciones_categoricas(var, one_hot))
        enviar = st.form_submit_button('Predecir consumo', type='primary', use_container_width=True)

    if enviar:
        if any(entrada.get(v) is None for v in cats):
            st.warning('Selecciona el origen del vehículo.')
            st.stop()
        X = preparar(entrada, one_hot, scaler, modelo)
        mpg = float(predecir(entrada, one_hot, scaler, modelo)[0])

        st.markdown('#### Resultado')
        tarjetas(mpg)
        medidor(mpg)

        ind = predicciones_individuales(modelo, X, scaler)
        if ind is not None and len(ind) > 1:
            p10, p90 = np.percentile(ind, [10, 90])
            st.info(f'El 80 % de los {len(ind)} estimadores del ensamble predicen entre **{p10:.1f} y {p90:.1f} mpg**.')

        with st.expander('Ver datos transformados que recibe el modelo'):
            st.write('Entrada original:'); st.dataframe(pd.DataFrame([entrada]), hide_index=True)
            st.write('Después de One-Hot + MinMaxScaler:'); st.dataframe(X.round(4), hide_index=True)

# =============================================================== pestaña 2: Excel
with tab_excel:
    requeridas = nums + cats
    st.markdown('#### Predicción para varios vehículos')
    st.markdown(f'Sube un archivo Excel (.xlsx) con una fila por vehículo y estas columnas: '
                + ', '.join(f'`{c}`' for c in requeridas) + '. Columnas adicionales se conservan en el resultado.')

    # plantilla de ejemplo
    ejemplo = pd.DataFrame([{v: RANGOS.get(v, {}).get('default', 0) for v in nums}
                            | {c: opciones_categoricas(c, one_hot)[0] for c in cats}] * 3)
    if 'origin' in ejemplo:
        ejemplo['origin'] = (opciones_categoricas('origin', one_hot) * 3)[:3]
    buf_pl = io.BytesIO(); ejemplo.to_excel(buf_pl, index=False)
    st.download_button('⬇ Descargar plantilla de Excel', buf_pl.getvalue(), 'plantilla_vehiculos.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

    archivo = st.file_uploader('Adjunta tu Excel', type=['xlsx', 'xls'])

    if st.button('Predecir desde Excel', type='primary', use_container_width=True, disabled=archivo is None):
        try:
            datos = pd.read_excel(archivo)
        except Exception as e:
            st.error(f'No se pudo leer el archivo: {e}')
            st.stop()

        claves = set(requeridas) | {'model_year'}
        datos = datos.rename(columns={c: str(c).strip().lower() for c in datos.columns
                                      if str(c).strip().lower() in claves})
        if 'vehicle_age' in requeridas and 'vehicle_age' not in datos and 'model_year' in datos:
            datos['vehicle_age'] = 2026 - datos['model_year']           # admite el año del modelo
        faltan = [c for c in requeridas if c not in datos.columns]
        if faltan:
            st.error('Faltan columnas en el Excel: ' + ', '.join(f'`{c}`' for c in faltan))
            st.stop()
        if 'origin' in datos:                                          # normaliza usa/Usa -> USA, etc.
            mapa = {o.lower(): o for o in opciones_categoricas('origin', one_hot)}
            datos['origin'] = datos['origin'].astype(str).str.strip().str.lower().map(mapa).fillna(datos['origin'])

        validas = datos[requeridas].notna().all(axis=1)
        resultado = datos.copy()
        resultado['mpg_predicho'] = np.nan
        if validas.any():
            resultado.loc[validas, 'mpg_predicho'] = predecir(datos.loc[validas, requeridas], one_hot, scaler, modelo).round(2)
        resultado['l_100km'] = (235.215 / resultado['mpg_predicho']).round(2)
        resultado['eficiencia'] = resultado['mpg_predicho'].apply(lambda m: nivel_eficiencia(m) if pd.notna(m) else 'sin datos')
        st.session_state['resultado_excel'] = resultado
        if (~validas).any():
            st.warning(f'{int((~validas).sum())} fila(s) con datos vacíos no se pudieron predecir.')

    if 'resultado_excel' in st.session_state:
        res = st.session_state['resultado_excel']
        ok = res['mpg_predicho'].dropna()
        if len(ok):
            st.markdown(f"""
            <div class="kpis">
              <div class="kpi"><div class="lbl">Vehículos</div><div class="val">{len(ok)}</div><div class="sub">predichos</div></div>
              <div class="kpi"><div class="lbl">Promedio</div><div class="val">{ok.mean():.1f} mpg</div><div class="sub">{235.215 / ok.mean():.1f} L/100 km</div></div>
              <div class="kpi"><div class="lbl">Rango</div><div class="val">{ok.min():.0f}–{ok.max():.0f}</div><div class="sub">mpg mínimo – máximo</div></div>
            </div>""", unsafe_allow_html=True)

            filtro = st.multiselect('Filtrar por eficiencia', ['alto', 'medio', 'bajo'], default=['alto', 'medio', 'bajo'])
            vista = res[res['eficiencia'].isin(filtro + ['sin datos'])]
            st.dataframe(vista, hide_index=True, use_container_width=True,
                         column_config={'mpg_predicho': st.column_config.ProgressColumn(
                             'mpg predicho', min_value=0, max_value=50, format='%.1f')})

            conteo = (res['eficiencia'].value_counts().reindex(['bajo', 'medio', 'alto']).fillna(0)
                      .rename_axis('eficiencia').reset_index(name='vehículos'))
            st.markdown('**Vehículos por nivel de eficiencia**')
            st.bar_chart(conteo, x='eficiencia', y='vehículos', color='#0F766E', horizontal=True,
                         height=170, sort=False)

            salida = io.BytesIO(); res.to_excel(salida, index=False)
            st.download_button('⬇ Descargar resultados en Excel', salida.getvalue(), 'predicciones_mpg.xlsx',
                               mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                               use_container_width=True)

st.markdown('<div class="footer">Proyecto de Machine Learning · Vehicle Fuel Consumption</div>', unsafe_allow_html=True)
