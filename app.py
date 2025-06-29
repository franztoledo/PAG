import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import random
import plotly.graph_objects as go
import heapq # Necesario para la cola de prioridad de Dijkstra

# --- 1. CLASES Y FUNCIONES DE ALGORITMOS MANUALES ---

# Estructura de datos Union-Find (o Disjoint Set Union) para Kruskal
class DSU:
    """Clase para la estructura de datos Union-Find, optimizada con unión por rango y compresión de caminos."""
    def __init__(self, nodes):
        self.parent = {node: node for node in nodes}
        self.rank = {node: 0 for node in nodes}

    def find(self, i):
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])  # Compresión de caminos
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Unión por rango/rango
            if self.rank[root_i] > self.rank[root_j]:
                self.parent[root_j] = root_i
            else:
                self.parent[root_i] = root_j
                if self.rank[root_i] == self.rank[root_j]:
                    self.rank[root_j] += 1
            return True
        return False

def kruskal_manual(graph, weight='tiempo'):
    """Implementación manual del algoritmo de Kruskal para encontrar el MST."""
    mst = nx.Graph()
    # 1. Obtener todas las aristas y ordenarlas por peso
    edges = sorted(graph.edges(data=True), key=lambda t: t[2].get(weight, 1))
    dsu = DSU(graph.nodes())
    
    for u, v, data in edges:
        # 2. Si añadir la arista no forma un ciclo (si u y v no están ya en el mismo conjunto)
        if dsu.find(u) != dsu.find(v):
            # 3. Unir los conjuntos y añadir la arista al MST
            dsu.union(u, v)
            mst.add_edge(u, v, **data)
    return mst

def dijkstra_manual(graph, start_node, weight='tiempo'):
    """
    Implementación manual del algoritmo de Dijkstra.
    Devuelve las distancias y los predecesores para reconstruir las rutas.
    """
    # 1. Inicialización
    distances = {node: float('inf') for node in graph.nodes()}
    predecessors = {node: None for node in graph.nodes()}
    distances[start_node] = 0
    
    # Cola de prioridad para almacenar (distancia, nodo)
    pq = [(0, start_node)]
    
    while pq:
        # 2. Obtener el nodo con la menor distancia actual
        current_distance, current_node = heapq.heappop(pq)
        
        # Si ya encontramos una ruta más corta, ignorar
        if current_distance > distances[current_node]:
            continue
            
        # 3. Explorar los vecinos
        for neighbor in graph.neighbors(current_node):
            edge_data = graph.get_edge_data(current_node, neighbor)
            edge_weight = edge_data.get(weight, 1)
            distance = current_distance + edge_weight
            
            # 4. Si encontramos una ruta más corta hacia el vecino
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                predecessors[neighbor] = current_node
                heapq.heappush(pq, (distance, neighbor))
                
    return distances, predecessors

def reconstruct_path(predecessors, start_node, end_node):
    """Reconstruye la ruta desde el diccionario de predecesores."""
    path = []
    current = end_node
    while current is not None:
        path.append(current)
        if current == start_node:
            break
        current = predecessors[current]
    # Si el bucle termina y el nodo inicial no está, no hay ruta
    return path[::-1] if start_node in path else None

# --- 2. FUNCIONES DE LÓGICA DE LA APLICACIÓN (ADAPTADAS) ---

@st.cache_data
def cargar_y_preparar_datos(nodos_path, aristas_path, centros_path):
    """Carga todos los datos desde archivos Excel y los normaliza."""
    nodos = pd.read_excel(nodos_path)
    aristas = pd.read_excel(aristas_path)
    centros = pd.read_excel(centros_path)
    
    # Normalizaciones de texto para consistencia
    nodos["centro_poblado"] = nodos["centro_poblado"].str.strip().str.upper()
    nodos["DISTRITO"] = nodos["DISTRITO"].str.strip().str.upper()
    aristas["origen"] = aristas["origen"].str.strip().str.upper()
    aristas["destino"] = aristas["destino"].str.strip().str.upper()
    centros["NOM_POBLAD"] = centros["NOM_POBLAD"].str.strip().str.upper()
    centros["DIST"] = centros["DIST"].str.strip().str.upper()
    
    return nodos, aristas, centros


@st.cache_resource
def crear_grafo_completo(_aristas_df):
    # ... (sin cambios)
    G = nx.Graph()
    for _, row in _aristas_df.iterrows():
        G.add_edge(row["origen"], row["destino"], tiempo=row["tiempo_minutos"], tipo=row["tipo_camino"])
    return G
def obtener_subgrafo_conectado(G_completo, nodos_df, aristas_df, centros_df):
    """
    Crea un subgrafo de muestra CONECTADO.
    Encuentra rutas entre nodos semilla y luego expande a sus distritos.
    """
    # 1. Identificar y clasificar capitales (igual que antes)
    capitales_df = centros_df[centros_df["CATEGORIA"] == "Capital de Distrito"].copy()
    lista_capitales = capitales_df["NOM_POBLAD"].unique().tolist()
    capital_a_distrito = pd.Series(capitales_df.DIST.values, index=capitales_df.NOM_POBLAD).to_dict()
    lista_hospitales = nodos_df[nodos_df["es_hospital"] == True]["centro_poblado"].unique().tolist()
    capitales_con_hospital = [cap for cap in lista_capitales if cap in lista_hospitales and cap in capital_a_distrito]
    capitales_sin_hospital = [cap for cap in lista_capitales if cap not in lista_hospitales and cap in capital_a_distrito]

    # 2. Validar que la muestra es posible
    if len(capitales_con_hospital) < 2 or len(capitales_sin_hospital) < 2:
        error_msg = f"No se pudo generar la muestra. Requisitos no cumplidos:\n- Capitales con hospital: {len(capitales_con_hospital)} (necesarias: 2)\n- Capitales sin hospital: {len(capitales_sin_hospital)} (necesarias: 2)"
        return None, None, error_msg

    # 3. Tomar la muestra aleatoria de nodos semilla
    capitales_seleccionadas = random.sample(capitales_con_hospital, 2) + random.sample(capitales_sin_hospital, 2)
    
    # 4. Crear la "columna vertebral" conectando los nodos semilla
    import itertools
    nodos_en_rutas = set(capitales_seleccionadas)
    
    # Pre-calcular predecesores para cada nodo semilla para optimizar
    predecesores_cache = {start_node: dijkstra_manual(G_completo, start_node)[1] for start_node in capitales_seleccionadas}

    for start_node, end_node in itertools.combinations(capitales_seleccionadas, 2):
        predecesores = predecesores_cache[start_node]
        path = reconstruct_path(predecesores, start_node, end_node)
        if path:
            nodos_en_rutas.update(path)

    # 5. Expandir a los distritos que tocan estas rutas
    distritos_finales = nodos_df[nodos_df['centro_poblado'].isin(nodos_en_rutas)]['DISTRITO'].unique().tolist()
    
    # 6. Crear el subgrafo final a partir de los distritos seleccionados
    nodos_sub = nodos_df[nodos_df["DISTRITO"].isin(distritos_finales)]
    poblados_sub = nodos_sub["centro_poblado"].tolist()
    aristas_sub = aristas_df[
        (aristas_df["origen"].isin(poblados_sub)) &
        (aristas_df["destino"].isin(poblados_sub))
    ]
    
    G_muestra = nx.Graph()
    for _, row in aristas_sub.iterrows():
        G_muestra.add_edge(row["origen"], row["destino"], tiempo=row["tiempo_minutos"], tipo=row["tipo_camino"])

    # Comprobación final de conectividad
    if not nx.is_connected(G_muestra):
        # Si aún no está conectado, toma el componente más grande
        componentes = sorted(nx.connected_components(G_muestra), key=len, reverse=True)
        G_muestra = G_muestra.subgraph(componentes[0]).copy()


    return G_muestra, distritos_finales, None


def calcular_ruta_dijkstra_wrapper(G, lista_hospitales, nodo_inicio):
    """
    Función "envoltorio" que usa la implementación manual de Dijkstra.
    """
    # Llama a nuestra función manual
    distances, predecessors = dijkstra_manual(G, nodo_inicio, weight='tiempo')
    
    hospital_mas_cercano = None
    tiempo_minimo = float('inf')
    
    for hospital in lista_hospitales:
        if hospital in distances and distances[hospital] < tiempo_minimo:
            tiempo_minimo = distances[hospital]
            hospital_mas_cercano = hospital
            
    if hospital_mas_cercano:
        # Llama a nuestra función para reconstruir la ruta
        ruta_optima = reconstruct_path(predecessors, nodo_inicio, hospital_mas_cercano)
        return ruta_optima, tiempo_minimo, hospital_mas_cercano
        
    return None, None, None

def visualizar_ruta_dijkstra_plotly(G, ruta_optima, nodo_inicio, hospital_mas_cercano):
    # ... (sin cambios)
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()
    # (Código de Plotly es largo, se omite por brevedad pero es el mismo de antes)
    edge_x_bg, edge_y_bg = [], []; path_edges = list(zip(ruta_optima, ruta_optima[1:]))
    for edge in G.edges():
        if edge not in path_edges and (edge[1], edge[0]) not in path_edges:
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_x_bg.extend([x0, x1, None]); edge_y_bg.extend([y0, y1, None])
    fig.add_trace(go.Scatter(x=edge_x_bg, y=edge_y_bg, line=dict(width=0.5, color='lightgray'), hoverinfo='none', mode='lines'))
    edge_x_path, edge_y_path = [], []
    for edge in path_edges:
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]; edge_x_path.extend([x0, x1, None]); edge_y_path.extend([y0, y1, None])
    fig.add_trace(go.Scatter(x=edge_x_path, y=edge_y_path, line=dict(width=4, color='black'), hoverinfo='none', mode='lines'))
    node_x, node_y, node_text, node_colors, node_sizes = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]; node_x.append(x); node_y.append(y); node_text.append(node)
        if node == nodo_inicio: node_colors.append('green'); node_sizes.append(20)
        elif node == hospital_mas_cercano: node_colors.append('red'); node_sizes.append(20)
        elif node in ruta_optima: node_colors.append('yellow'); node_sizes.append(15)
        else: node_colors.append('lightgray'); node_sizes.append(8)
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers', text=node_text, hoverinfo='text', marker=dict(color=node_colors, size=node_sizes, line=dict(width=1, color='black'))))
    fig.update_layout(title_text=f'<b>Ruta Óptima de {nodo_inicio} a {hospital_mas_cercano}</b>', title_x=0.5, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), margin=dict(l=20, r=20, t=40, b=20))
    return fig
def visualizar_mst_plotly(G, MST, lista_hospitales, lista_capitales):
    """
    Crea una figura de Plotly para el MST, coloreando hospitales y capitales.
    """
    pos = nx.spring_layout(G, seed=42)
    fig = go.Figure()

    # --- Trazas de Aristas (Edges) ---
    # Aristas de fondo (no en el MST)
    edge_x_bg, edge_y_bg = [], []
    mst_edges = set(map(frozenset, MST.edges()))
    for edge in G.edges():
        if frozenset(edge) not in mst_edges:
            x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
            edge_x_bg.extend([x0, x1, None]); edge_y_bg.extend([y0, y1, None])
    fig.add_trace(go.Scatter(x=edge_x_bg, y=edge_y_bg, line=dict(width=1, color='lightgray', dash='dash'),
        hoverinfo='none', mode='lines', name='Caminos No Utilizados'))

    # Aristas del MST
    edge_x_mst, edge_y_mst = [], []
    for edge in MST.edges():
        x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
        edge_x_mst.extend([x0, x1, None]); edge_y_mst.extend([y0, y1, None])
    fig.add_trace(go.Scatter(x=edge_x_mst, y=edge_y_mst, line=dict(width=3, color='red'),
        hoverinfo='none', mode='lines', name='Red de Conexión Mínima (MST)'))

    # --- Traza de Nodos con Colores Específicos ---
    node_x, node_y, node_colors, node_sizes = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Lógica de coloreado con prioridad para hospitales
        if node in lista_hospitales:
            node_colors.append('red')
            node_sizes.append(12)
        elif node in lista_capitales:
            node_colors.append('blue')
            node_sizes.append(12)
        elif node in MST.nodes():
            node_colors.append('#606060') # Gris oscuro para nodos regulares en el MST
            node_sizes.append(8)
        else:
            node_colors.append('lightgray') # Gris claro para nodos de fondo
            node_sizes.append(4)
            
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='none',
        marker=dict(color=node_colors, size=node_sizes, line=dict(width=1, color='black')),
        showlegend=False
    ))

    # --- Configuración del Layout ---
    fig.update_layout(
        title_text='<b>Red de Conexión Mínima (MST)</b>', title_x=0.5,
        showlegend=True, legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20), plot_bgcolor='white'
    )
    return fig


def calcular_y_visualizar_mst_wrapper(G):
    """
    Función "envoltorio" que usa la implementación manual de Kruskal
    y la nueva visualización de Plotly.
    """
    st.header("Red de conexión mínima (Implementación Propia de Kruskal)")
    st.write("Visualización de la red de caminos más corta que conecta todas las comunidades.")

    # 1. Calcular el MST (sin cambios)
    MST = kruskal_manual(G, weight='tiempo')
    total_tiempo_mst = MST.size(weight='tiempo')
    st.info(f"El tiempo total mínimo para conectar todas las comunidades es: **{total_tiempo_mst:.2f} minutos**")

    # 2. Visualizar con la nueva función de Plotly
    fig = visualizar_mst_plotly(G, MST)
    st.plotly_chart(fig, use_container_width=True)

# --- 3. APLICACIÓN PRINCIPAL DE STREAMLIT ---
def main():
    """Función principal que ejecuta la aplicación de Streamlit."""
    
    st.set_page_config(page_title="Análisis de Redes Viales", layout="wide")
    st.title("Análisis de Redes de Centros Poblados")

    # Carga de todos los datos necesarios
    try:
        nodos_df, aristas_df, centros_df = cargar_y_preparar_datos(
            nodos_path="nodos_OBTENIDOS_actualizado.xlsx",
            aristas_path="aristas_OBTENIDOS.xlsx",
            centros_path="CENTROSPOBLADOS.xlsx"
        )
        G_completo = crear_grafo_completo(aristas_df)
    except FileNotFoundError as e:
        st.error(f"Error: No se pudo encontrar el archivo de datos: {e.filename}. Asegúrate de que los archivos Excel estén en la misma carpeta.")
        return

    # Generar listas de nodos especiales para reutilizarlas
    lista_hospitales = nodos_df[nodos_df["es_hospital"] == True]["centro_poblado"].unique().tolist()
    capitales_df = centros_df[centros_df["CATEGORIA"] == "Capital de Distrito"]
    lista_capitales = capitales_df["NOM_POBLAD"].unique().tolist()

    # Barra lateral
    st.sidebar.header("Configuración")
    opciones_analisis = [
        "Ruta más rápida a Hospital (Dijkstra)", 
        "MST de la Red Completa (Kruskal)",
        "MST sobre Muestra de Distritos"
    ]
    analisis_seleccionado = st.sidebar.selectbox(
        "Elige el análisis que quieres realizar:", opciones_analisis
    )

    # --- Lógica para cada sección ---

    if analisis_seleccionado == "Ruta más rápida a Hospital (Dijkstra)":
        st.header("Encontrar la ruta más rápida a un hospital (Implementación Propia de Dijkstra)")
        lista_nodos_validos = sorted([nodo for nodo in G_completo.nodes()])
        default_index = lista_nodos_validos.index("LIMA") if "LIMA" in lista_nodos_validos else 0
        nodo_inicio = st.selectbox("Selecciona el centro poblado de origen:", lista_nodos_validos, index=default_index)
        
        if st.button("Calcular Ruta"):
            ruta_optima, tiempo_minimo, hospital_mas_cercano = calcular_ruta_dijkstra_wrapper(G_completo, lista_hospitales, nodo_inicio)
            if ruta_optima:
                st.success(f"Hospital más cercano: **{hospital_mas_cercano}**")
                st.info(f"Tiempo de viaje estimado: **{tiempo_minimo:.2f} minutos**")
                fig_dijkstra = visualizar_ruta_dijkstra_plotly(G_completo, ruta_optima, nodo_inicio, hospital_mas_cercano)
                st.plotly_chart(fig_dijkstra, use_container_width=True)
                st.markdown("""
                **Leyenda de Nodos:**
                - <span style="color:green; font-size: 20px;">●</span> **Nodo de Partida**
                - <span style="color:red; font-size: 20px;">●</span> **Hospital Destino**
                - <span style="color:yellow; font-size: 20px;">●</span> **Nodos en Ruta**
                """, unsafe_allow_html=True)
            else:
                st.error(f"No se encontró una ruta desde '{nodo_inicio}' a ningún hospital.")


    elif analisis_seleccionado == "MST de la Red Completa (Kruskal)":
        st.header("MST de la Red Completa (Implementación Propia de Kruskal)")
        MST_completo = kruskal_manual(G_completo, weight='tiempo')
        st.info(f"Tiempo total para conectar la red completa: **{MST_completo.size(weight='tiempo'):.2f} minutos**")
        
        fig_mst_completo = visualizar_mst_plotly(G_completo, MST_completo, lista_hospitales, lista_capitales)
        st.plotly_chart(fig_mst_completo, use_container_width=True)
        
        # --- LEYENDA AÑADIDA ---
        st.markdown("""
        **Leyenda de Nodos:**
        - <span style="color:red; font-size: 20px;">●</span> **Hospital**
        - <span style="color:blue; font-size: 20px;">●</span> **Capital de Distrito**
        - <span style="color:#606060; font-size: 20px;">●</span> **Otro Nodo en el MST**
        - <span style="color:lightgray; font-size: 16px;">●</span> *Otro Nodo (fuera del MST)*
        """, unsafe_allow_html=True)


    elif analisis_seleccionado == "MST sobre Muestra de Distritos":
        st.header("MST sobre una Muestra Aleatoria y Conectada de Distritos")
        
        if st.button("Generar Nueva Muestra Conectada"):
            pass

        G_muestra, distritos_muestra, error = obtener_subgrafo_conectado(G_completo, nodos_df, aristas_df, centros_df)

        if error:
            st.error(error)
        else:
            st.success(f"Muestra conectada generada para los distritos: **{', '.join(distritos_muestra)}**")
            MST_muestra = kruskal_manual(G_muestra, weight='tiempo')
            st.info(f"El tiempo total mínimo para conectar esta muestra es: **{MST_muestra.size(weight='tiempo'):.2f} minutos**")
            
            fig_mst_muestra = visualizar_mst_plotly(G_muestra, MST_muestra, lista_hospitales, lista_capitales)
            st.plotly_chart(fig_mst_muestra, use_container_width=True)

            # --- LEYENDA AÑADIDA ---
            st.markdown("""
            **Leyenda de Nodos:**
            - <span style="color:red; font-size: 20px;">●</span> **Hospital**
            - <span style="color:blue; font-size: 20px;">●</span> **Capital de Distrito**
            - <span style="color:#606060; font-size: 20px;">●</span> **Otro Nodo en el MST**
            - <span style="color:lightgray; font-size: 16px;">●</span> *Otro Nodo (fuera del MST)*
            """, unsafe_allow_html=True)
            
if __name__ == "__main__":
    main()