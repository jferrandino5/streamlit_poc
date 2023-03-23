#todo: update so that connections with heighest weighted edges are returned
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('archive/member-edges.csv')
df_member = pd.read_csv('archive/meta-members.csv')
df_D = df[0 : 1000]
#set up graph
G = nx.from_pandas_edgelist(df, 'member1', 'member2', edge_attr='weight', create_using = nx.Graph())
node_attr = df_member.set_index('member_id').to_dict('index')
nx.set_node_attributes(G, node_attr)
#grab a smaller graph for speed
G_D = nx.from_pandas_edgelist(df_D, 'member1', 'member2', edge_attr='weight', create_using = nx.Graph())
node_attr = df_member.set_index('member_id').to_dict('index')
nx.set_node_attributes(G_D, node_attr)

val_map = {'member': 0}

# Define function to get the number of connected nodes and names of characters most connected to the input character
def get_connection_info(name):
    #todo handle when name is not in graph
    try:
        node_id= [x for x, y in G.nodes(data=True) if len(y) >0 and y['name'] == name][0]
        if node_id in G:
            # Get the number of connected nodes
            num_connections = len(G[node_id])
            # Get the names of characters most connected to the input character
            connected_characters = sorted(G[node_id], key=lambda x: G.degree(x), reverse=True)[:10]
            connected_names = [G.nodes[c]['name'] for c in connected_characters if G.nodes[c]['name'] != 'nan']

            return num_connections, connected_names
        else:
            return None, None
    except:
        return None, None
#iterate two nodes out to get 2 hop away neighbors
def get_connected_nodes(name, graph):
    connected_nodes = {}
    node_id = [x for x, y in G.nodes(data=True) if len(y) >0 and y['name'] == name][0]
    # Get the neighbors of the selected character
    neighbors = list(G.neighbors(node_id))
    # Iterate through the neighbors to find their neighbors
    for neighbor in neighbors:
        # Get the neighbors of the neighbor
        sub_neighbors = list(G.neighbors(neighbor))
        sub_neighbor_names = [G.nodes[node]['name'] for node in sub_neighbors if graph.nodes[node]['name'] != 'nan']
        connected_nodes[G.nodes[neighbor]['name']] = sub_neighbor_names
    return connected_nodes

#get node from input name
def search_attribute(search_value, attribute, g):
    node_id = [x for x, y in g.nodes(data=True) if len(y[attribute]) > 0 if y[attribute] == search_value][0]
    return node_id

#find neighbors K number of edges away
def k_neighbors(G, node, k):
    subgraph = nx.ego_graph(G, node, radius=k)
    neighbors = list(subgraph.nodes())
    return neighbors

#get degree info
def get_first_degree(graph,name):
    node_id = [x for x, y in graph.nodes(data=True) if len(y) >0 and y['name'] == name][0]
    path_lengths = nx.single_source_dijkstra_path_length(graph, node_id)
    key_list = []
    for key, value in path_lengths.items():
        if value ==1:
            key_list.append(key)

    edge_list = []
    for i in graph.edges():
        if (i[0] == node_id or i[1] == node_id) and (i[0] in key_list or i[1] in key_list):
            edge_list.append(i)
    return edge_list

#predict which nodes have a possible connection to the entry node
#there are many metrics for link prediction that could contribute or train a model on the data 
def get_link_predict(name,graph):
    node_id =[x for x, y in graph.nodes(data=True) if len(y) >0 and y['name'] == name][0]
    link_predict = [list(nx.jaccard_coefficient(graph,[(node_id, x)])) for x, y in graph.nodes(data=True)]
    #sorted_list = sorted(link_predict, key=itemgetter(2),reverse=True)
    #set threshold 
    link_predict_filter = [x[0][0:2] for x in link_predict if x[0][2]>.1 and x[0][2]<1]
    link_predict_name = [graph.nodes[x[1]]['name'] for x in link_predict_filter][1:]
    return link_predict,link_predict_filter, link_predict_name

#populate the graph with actual and possible neighbors
def get_graph(edge_list):
    G_sub = nx.Graph()
    G_sub.add_edges_from(edge_list,color='b')
    edges = G_sub.edges()
    colors_sub = [G_sub[u][v]['color'] for u,v in edges]
    G_sub_neighbors = list(G_sub.nodes())
    FG = nx.Graph()
    FG.add_edges_from(link_predict_filter)
    unique_edge = []
    for i in FG.edges():
        if i not in G_sub.edges():
            unique_edge.append(i)
    add = nx.Graph()
    add.add_edges_from(unique_edge,color='r')
    edges = add.edges()
    colors_add = [add[u][v]['color'] for u,v in edges]
    add_neighbors = list(add.nodes())
    F = nx.compose(G_sub,add)
    F_neighbors = list(F.nodes())
    return F, G_sub_neighbors, add_neighbors, F_neighbors, colors_sub, colors_add
   
#streamlit functions
def list_related_information(input_list, attribute):
    if input_list:
        st.write(f"Related {attribute} are:")
        for i, value in enumerate(input_list):
            st.write(f"{i + 1}. {value}")
# Set up the Streamlit app
st.title("Social Network Query")

# Get user input for character name
search_name = st.text_input("Enter the name of a entity to search:", value ="")

#draw the full graph if no character name is entered
if search_name== '':
    # Draw the network graph of the input character and its connections
    color_map = [val_map.get('member', 0.25) for node, attributes in G_D.nodes(data=True)]

    #dictionary with nodes as keys and table as the value
    node_table_labels = {node: 'member' for node, attributes in G_D.nodes(data=True)}

    nodes = G_D.nodes()
    degree = G_D.degree()
    colors = [degree[n] for n in nodes]
    #size = [(degree[n]) for n in nodes]

    pos = nx.kamada_kawai_layout(G_D)
    #pos = nx.spring_layout(G, k = 0.2)
    cmap = plt.cm.viridis_r
    cmap = plt.cm.Greys

    vmin = min(colors)
    vmax = max(colors)

    fig, ax = plt.subplots(figsize = (15,9), dpi=100)
    nx.draw(G_D,pos,ax=ax,alpha = 0.8, nodelist = nodes, node_color = 'w', node_size = 10, with_labels= False,font_size = 6, width = 0.2, cmap = cmap, edge_color ='yellow')
    fig.set_facecolor('#0B243B')
    ax.set_title(f"Full Network Graph")
    st.pyplot(fig)
    st.stop()

if search_name != "":

    # Call the function to get connection information for the input character
    num_connections, connected_character_names = get_connection_info(search_name)
    # parties_2_edges_away = get_connected_nodes(character_name, G)
    result_node_id = search_attribute(search_name, 'name', G_D)
    # Display the connection information

if num_connections is not None and connected_character_names is not None:
    st.write(f"{search_name} is connected to {num_connections} other nodes.")
    st.write(f"The entities most connected to {search_name} are:")
    for i, name in enumerate(connected_character_names):
        st.write(f"{i + 1}. {name}")

    # node_id = [x for x, y in G.nodes(data=True) if y['name'] == name][0]

    # Draw the network graph of the input character and its connections
    subgraph = nx.ego_graph(G, result_node_id, radius=1)
    link_predict,link_predict_filter, link_predict_name = get_link_predict(name,G)
    edge_list = get_first_degree(G,name)
    F, G_sub_neighbors, add_neighbors, F_neighbors,colors_sub, colors_add = get_graph(edge_list)

    fig, ax = plt.subplots(figsize=(6, 6))
    nodes = F.nodes()
    degree = F.degree()
    colors = colors_sub+colors_add
    pos = nx.kamada_kawai_layout(F)
    cmap = plt.cm.viridis_r
    cmap = plt.cm.Greys

    color_map = [val_map.get('member', 0.25) for node, attributes in subgraph.nodes(data=True)]

    #dictionary with nodes as keys and table as the value
    node_table_labels = {node: 'member' + '\n' + attributes['name']
                         for node, attributes in subgraph.nodes(data=True)}

    nx.draw(F,pos,alpha = 0.8, nodelist = nodes, node_color = 'black', node_size = 10, with_labels= True,font_size = 6, width = 0.2, 
        cmap = cmap, edge_color = colors)
    ax.set_title(f"Network Graph for {search_name} and Their Connections")
    st.pyplot(fig)

    # if parties_2_edges_away:
    #     st.write(f"The characters 2 edges away from {character_name} are:")
    #     for i, party in enumerate(parties_2_edges_away.items()):
    #         st.write(f"{i + 1}. {party[0]}")
else:
    st.write("Please enter a valid entity name.")