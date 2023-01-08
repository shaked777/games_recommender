from IPython.display import display
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def main():
  plt.style.use('bmh')

  # The data
  data = pd.read_csv('/Users/shakedgabay/Desktop/python/python_course/reccomandition_system/video_games.csv')
  data = data.rename(columns={'plot': 'description'})
  print(data.columns)
  print(type(data))
  X = np.array(data.description)


  # View the data
  data = data[['name','description','rating', 'url']].astype(str)
  a = data.head()
  # display(a)

  # Translate the data to numbers(vectors) BERT
  text_data = X
  model = SentenceTransformer('distilbert-base-nli-mean-tokens')
  embeddings = model.encode(text_data, show_progress_bar=True)
  embed_data = embeddings

  # PCA
  X = np.array(embed_data)
  n_comp = 5
  pca = PCA(n_components=n_comp)
  pca.fit(X)
  pca_data = pd.DataFrame(pca.transform(X))
  b = pca_data.head()


  # Give reccomendatio by the cosine simalrity of the vectors
  cos_sim_data = pd.DataFrame(cosine_similarity(X))
  def give_recommendations(index,print_recommendation = False):
    index_recomm =cos_sim_data.loc[index].sort_values(ascending=False).index.tolist()[1:4]
    games_recomm =  data['name'].loc[index_recomm].values
    result = {'Games':games_recomm,'Index':index_recomm}
    if print_recommendation==True:
      print('The game choosen is: %s \n'%(data['name'].loc[index]))
      k=1
      for game in games_recomm:
        print('The number %i recommended game is: %s \n'%(k,game))
        locate = data.loc[data['name']==str(game)].index[0]
        print('The rating of that game is: \n'+data['rating'].loc[locate])
        print("\n")
        print('Check it out: \n'+data['url'].loc[locate])
        print("\n")
        print('%s\n'%(data['description'].loc[(int(data.loc[data['name']==str(game)].index[0]))]))
        k=k+1
        print("--------------------------------------------------")
    return result


# 0	Spider-Man
# 1	Red Dead Redemption II
# 2	Grand Theft Auto V
# 3	God of War
# 4	Uncharted 4: A Thief's End
# 5	The Last of Us: Part II
# 6	Horizon Forbidden West
# 7	The Last of Us
# 8	Detroit: Become Human
# 9	Death Stranding
# 10	Ghost of Tsushima
# Give recommendations the index above (full list at csv file)
  index_r = 2
  give_recommendations(index_r,True)

  # Plot random reccomendation
  plt.figure(figsize=(20,20))

  plt.subplot()
  index = index_r
  to_plot_data = cos_sim_data.drop(index,axis=1)
  plt.plot(to_plot_data.loc[index],'.',color='firebrick')
  recomm_index = give_recommendations(index)
  x = recomm_index['Index']
  y = cos_sim_data.loc[index][x].tolist()
  m = recomm_index['Games']
  plt.plot(x,y,'.',color='navy',label='Recommended games')
  plt.title('Game Liked: '+data['name'].loc[index])
  plt.xlabel('Game Index')
  k=0
  for x_i in x:
      plt.annotate('%s'%(m[k]),(x_i,y[k]),fontsize=10)
      k=k+1

  plt.ylabel('Cosine Similarity')
  plt.ylim(0,1)
  plt.show()

if __name__ == "__main__":
    main()

