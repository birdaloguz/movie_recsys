import re, scipy, numpy as np, requests, pandas as pd
from quiz.recsys import csr_matrix_indices
from scipy import sparse
from quiz.theano_bpr import BPR
from six.moves import cPickle
import os
from scipy.sparse.linalg import svds
from quiz.clustering import clustering
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import shuffle

from quiz.recsys import bpr, knn, matrix_factorization
from sklearn.model_selection import train_test_split

df_movies_org = pd.read_csv('/home/binglidev001/movie_recsys/dataset/movietweetings/movies.dat', sep='::', header=None, names=["movie_id", "title", "genre"])
df_ratings_org = pd.read_csv('/home/binglidev001/movie_recsys/dataset/movietweetings/ratings.dat', sep='::', header=None, names=["user_id", "movie_id", "rating", "timestamp"])


triplets = clustering(df_ratings_org)
triplets = pd.DataFrame(triplets, columns=[1, 2, 3])

df_movies_org = df_movies_org[df_movies_org["movie_id"].isin(df_ratings_org.movie_id.unique())].reset_index()

# create movie-ratings matrix
matrix_df = df_ratings_org.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
matrix_df_train, matrix_df_test = train_test_split(matrix_df, test_size=0.2)
print(matrix_df_train)
print(matrix_df_test)


um_matrix = scipy.sparse.csr_matrix(matrix_df_train.transpose().values)

# matrix factorization model
um_matrix_mf = scipy.sparse.csr_matrix(matrix_df_train.values)

movie_columns = matrix_df_train.transpose().columns
user_ratings_mean = np.mean(um_matrix_mf, axis=1)
R_demeaned = um_matrix_mf  # - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

# knn model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=6, n_jobs=-1)
model_knn.fit(um_matrix)


#load existing bpr model
f = open(os.path.dirname(os.path.abspath(__file__)) + '/models/bpr_model.save', 'rb')
bpr_model = cPickle.load(f)
f.close()




correlation_results=[]
for idx, row in matrix_df_test.iterrows():

    # choose movies to offer from triplets
    offered_movies = []
    offered_hist_movies = []
    movies_to_offer = triplets.sample(n=40)
    # movies_to_offer = triplets.head(30)
    for index, row in movies_to_offer.iterrows():
        triplet = [row[1], row[2], row[3]]
        triplet = shuffle(triplet)
        if triplet[0] not in offered_movies and triplet[0] not in offered_hist_movies:
            offered_movies.append(triplet[0])
        if triplet[1] not in offered_hist_movies and triplet[1] not in offered_movies:
            offered_hist_movies.append(triplet[1])
        if triplet[2] not in offered_hist_movies and triplet[2] not in offered_movies:
            offered_hist_movies.append(triplet[2])

    offered_movies = offered_movies[:20]
    offered_hist_movies = offered_hist_movies[:60]

    # get dataframes of movies will be offered
    offered_movies = df_movies_org.loc[df_movies_org["movie_id"].isin(offered_movies)]

    offered_hist_movies = df_movies_org.loc[df_movies_org["movie_id"].isin(offered_hist_movies)]

    offered_top = [row['movie_id'] for idx, row in offered_movies.iterrows()]
    hist_user = [str(row['movie_id']) for idx, row in offered_hist_movies.head(10).iterrows()]

    print(offered_top)
    results1 = matrix_factorization(hist_user, offered_top, df_movies_org, df_ratings_org, U, sigma, Vt, movie_columns)
    print(results1)
    results2 = knn(hist_user, offered_top, matrix_df, um_matrix, model_knn, movie_columns, df_movies_org)
    print(results2)
    results3 = bpr(hist_user, offered_top, df_movies_org, df_ratings_org, bpr_model, matrix_df.index)
    print(results3)

    # calculate the spearman's correlation between two variables
    from scipy.stats import spearmanr
    # calculate spearman's correlation
    coef1, p1 = spearmanr(results1, results2)
    print('Spearmans correlation coefficient: %.3f' % coef1)
    # interpret the significance
    alpha = 0.05
    if (p1 > alpha):
    	print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p1)
    else:
    	print('Samples are correlated (reject H0) p=%.3f' % p1)

    # calculate spearman's correlation
    coef2, p2 = spearmanr(results1, results3)
    print('Spearmans correlation coefficient: %.3f' % coef2)
    # interpret the significance
    alpha = 0.05
    if (p2 > alpha):
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p2)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p2)

    # calculate spearman's correlation
    coef3, p3 = spearmanr(results2, results3)
    print('Spearmans correlation coefficient: %.3f' % coef3)
    # interpret the significance
    alpha = 0.05
    if (p3 > alpha):
        print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p3)
    else:
        print('Samples are correlated (reject H0) p=%.3f' % p3)

    correlation_results.append([[coef1, p1], [coef2, p2], [coef3, p3]])

print(correlation_results)

correlation_results = [[[0.30075187969924805, 0.19757966317255538], [-0.051127819548872175, 0.8304946161837171], [0.18646616541353384, 0.43119491043425096]], [[0.6225563909774435, 0.003372243843155665], [-0.004511278195488721, 0.9849400483404018], [-0.13684210526315788, 0.5650957921363965]], [[0.30225563909774433, 0.1952375807222028], [0.10676691729323308, 0.6541415450094847], [0.15939849624060148, 0.5020524593517314]], [[0.07218045112781954, 0.7623392575365604], [0.12932330827067667, 0.586852724489982], [-0.20751879699248119, 0.37999323699324683]], [[0.060150375939849614, 0.8011125555760907], [-0.0015037593984962405, 0.9949797739432688], [-0.2120300751879699, 0.36948599387790193]], [[0.2421052631578947, 0.30375506301143457], [-0.15037593984962402, 0.5268549482529812], [0.38195488721804516, 0.09653857766782038]], [[-0.09774436090225563, 0.681832438507082], [-0.14285714285714285, 0.5479501508336087], [0.40751879699248117, 0.07450821585547025]], [[0.13383458646616542, 0.5737559530736434], [-0.156390977443609, 0.5102566899029022], [0.09774436090225563, 0.681832438507082]], [[0.5473684210526315, 0.012490190369603566], [0.2781954887218045, 0.23497379059889356], [0.03308270676691729, 0.889875964885619]], [[-0.051127819548872175, 0.8304946161837171], [-0.12932330827067667, 0.586852724489982], [-0.22857142857142856, 0.3323968570405118]], [[0.030075187969924807, 0.8998365956859304], [-0.22255639097744356, 0.3456206337468547], [-0.2706766917293233, 0.24839056913669727]], [[-0.2150375939849624, 0.36257388702455096], [0.3954887218045112, 0.0843516324500122], [-0.06466165413533834, 0.7865148626867856]], [[0.23759398496240602, 0.3131305637442674], [-0.22105263157894736, 0.3489737543021465], [-0.037593984962406006, 0.8749652018566716]], [[0.3533834586466165, 0.12640665077932206], [0.14736842105263157, 0.5352473458918693], [-0.007518796992481202, 0.9749025019933399]], [[-0.04661654135338346, 0.8452698052894374], [-0.23609022556390977, 0.3162939707830972], [-0.18796992481203004, 0.42742125802266384]], [[0.27518796992481204, 0.2402830280811375], [-0.2902255639097744, 0.21449998226074232], [-0.1639097744360902, 0.4898669433807742]], [[0.2571428571428572, 0.27375050350001584], [0.4300751879699248, 0.05839600708511875], [0.1609022556390977, 0.4979744144247943]], [[0.1533834586466165, 0.5185244542213634], [0.1969924812030075, 0.40515233009814566], [-0.04210526315789473, 0.8600949015145793]], [[0.24360902255639094, 0.3006681944004437], [-0.2120300751879699, 0.36948599387790193], [0.29624060150375936, 0.20471829742862183]], [[0.41353383458646614, 0.06992073829262123], [0.5503759398496241, 0.011919637001913333], [0.03609022556390977, 0.8799311955207728]], [[-0.05714285714285714, 0.8108800461568323], [0.037593984962406006, 0.8749652018566716], [0.193984962406015, 0.4125038246613568]], [[-0.29172932330827067, 0.21202623258054837], [-0.43458646616541347, 0.05551879139596901], [0.16992481203007515, 0.47384906489411216]], [[0.009022556390977442, 0.9698849998123449], [0.012030075187969924, 0.9598534442846784], [0.3157894736842105, 0.17499413541563488]], [[0.26315789473684204, 0.2622871228754285], [0.015037593984962403, 0.9498276950483329], [0.02706766917293233, 0.9098116651774512]], [[0.06466165413533834, 0.7865148626867856], [0.10827067669172932, 0.6495672222012883], [-0.4330827067669173, 0.05646561004225937]], [[-0.19849624060150375, 0.4015036070038971], [-0.1413533834586466, 0.5522144386322418], [0.42556390977443603, 0.06138492007540436]], [[0.019548872180451125, 0.9348031235214638], [0.24360902255639094, 0.3006681944004437], [-0.4060150375939849, 0.0756893721143978]], [[0.30977443609022554, 0.18380643241191041], [-0.05864661654135337, 0.8059928433862233], [0.12030075187969923, 0.6134176645307821]], [[-0.28421052631578947, 0.22458459821120463], [0.3218045112781954, 0.166474348578758], [-0.10075187969924812, 0.6725564354857818]], [[0.29172932330827067, 0.21202623258054837], [0.31729323308270674, 0.1728368864071423], [0.44060150375939844, 0.05185183693270709]], [[0.04360902255639097, 0.8551479470173743], [-0.07669172932330826, 0.7479334475419401], [0.16240601503759397, 0.4939125443630319]], [[0.625563909774436, 0.003178354611178857], [-0.045112781954887216, 0.8502061902084872], [0.030075187969924807, 0.8998365956859304]], [[-0.018045112781954885, 0.9398091995407436], [-0.010526315789473682, 0.9648685868194993], [0.26766917293233083, 0.2538915426582106]], [[-0.2827067669172932, 0.22715328510473623], [0.051127819548872175, 0.8304946161837171], [-0.12330827067669171, 0.6045088306268851]], [[-0.2827067669172932, 0.22715328510473623], [0.015037593984962403, 0.9498276950483329], [0.09473684210526315, 0.6911525963694354]], [[-0.09473684210526315, 0.6911525963694354], [-0.07218045112781954, 0.7623392575365604], [0.18345864661654135, 0.4387948865691028]], [[0.1518796992481203, 0.5226819111403139], [-0.1894736842105263, 0.42366525592686277], [-0.27518796992481204, 0.2402830280811375]], [[-0.3609022556390977, 0.11798037156435756], [0.03909774436090225, 0.8700037043301221], [-0.3142857142857143, 0.17716966765175673]], [[0.051127819548872175, 0.8304946161837171], [0.06917293233082707, 0.7719854265259412], [0.4676691729323308, 0.03758851020414889]], [[-0.35639097744360904, 0.12298631080226882], [0.2526315789473684, 0.2825500572639195], [-0.030075187969924807, 0.8998365956859304]], [[0.09022556390977443, 0.7052130308607131], [0.08270676691729321, 0.7288508763559796], [0.037593984962406006, 0.8749652018566716]], [[0.4541353383458646, 0.04427910416289474], [-0.03458646616541353, 0.8849015088305833], [0.193984962406015, 0.4125038246613568]], [[-0.27518796992481204, 0.2402830280811375], [0.5669172932330826, 0.00914627004759854], [-0.1263157894736842, 0.5956535128148985]], [[-0.07218045112781954, 0.7623392575365604], [-0.26315789473684204, 0.2622871228754285], [-0.16992481203007515, 0.47384906489411216]], [[0.2857142857142857, 0.22203494893940093], [0.27368421052631575, 0.24296636906130623], [-0.0706766917293233, 0.7671582206450904]], [[-0.12481203007518797, 0.6000744184144979], [-0.037593984962406006, 0.8749652018566716], [-0.060150375939849614, 0.8011125555760907]], [[-0.05263157894736842, 0.825581450455932], [-0.22255639097744356, 0.3456206337468547], [0.2270676691729323, 0.3356744310134179]], [[0.21353383458646616, 0.36602063833951404], [0.27368421052631575, 0.24296636906130623], [0.16240601503759397, 0.4939125443630319]], [[0.2045112781954887, 0.38709023775567686], [0.024060150375939848, 0.9197997451459456], [-0.07368421052631578, 0.757528699864343]], [[0.45714285714285713, 0.042718510627876995], [0.2721804511278195, 0.24566887828037737], [-0.2150375939849624, 0.36257388702455096]], [[-0.08421052631578946, 0.7241036435108112], [-0.21654135338345865, 0.3591457825704002], [0.022556390977443608, 0.9247982163788796]], [[0.24962406015037594, 0.28851258969719185], [-0.10526315789473684, 0.6587277693915812], [0.08421052631578946, 0.7241036435108112]], [[0.08421052631578946, 0.7241036435108112], [-0.17593984962406017, 0.45809865870248545], [0.10375939849624059, 0.6633257542362446]], [[0.29774436090225564, 0.20231998601897422], [0.12781954887218044, 0.5912462398946646], [0.12330827067669171, 0.6045088306268851]], [[0.2150375939849624, 0.36257388702455096], [0.08721804511278194, 0.7146383646582315], [0.015037593984962403, 0.9498276950483329]], [[0.12030075187969923, 0.6134176645307821], [0.13984962406015036, 0.5564935505662203], [-0.3082706766917293, 0.1860555607781014]], [[0.2421052631578947, 0.30375506301143457], [0.27518796992481204, 0.2402830280811375], [0.09172932330827067, 0.7005157258604466]], [[-0.17593984962406017, 0.45809865870248545], [0.024060150375939848, 0.9197997451459456], [-0.16691729323308271, 0.48182491843154673]], [[0.11879699248120301, 0.6178918280262935], [-0.5052631578947367, 0.023059229474174193], [-0.3443609022556391, 0.13707258150434384]], [[0.09624060150375939, 0.6864870716570238], [-0.20601503759398496, 0.3835325530925867], [-0.08872180451127819, 0.7099206278927328]], [[-0.06165413533834586, 0.7962393511384775], [-0.21954887218045113, 0.352345677311515], [-0.10375939849624059, 0.6633257542362446]], [[0.3278195488721804, 0.15824386826406367], [0.06917293233082707, 0.7719854265259412], [0.3278195488721804, 0.15824386826406367]], [[0.0781954887218045, 0.7431490742332265], [-0.3203007518796992, 0.1685770430646261], [0.019548872180451125, 0.9348031235214638]], [[0.1894736842105263, 0.42366525592686277], [-0.2601503759398496, 0.26798033372196073], [0.06616541353383458, 0.7816639121085741]], [[0.45112781954887216, 0.04588325880000641], [0.004511278195488721, 0.9849400483404018], [0.16992481203007515, 0.47384906489411216]], [[0.007518796992481202, 0.9749025019933399], [0.156390977443609, 0.5102566899029022], [-0.13233082706766916, 0.5781074593622941]], [[0.21804511278195485, 0.35573636611773685], [0.1533834586466165, 0.5185244542213634], [-0.193984962406015, 0.4125038246613568]], [[0.18045112781954886, 0.4464646102705483], [0.21353383458646616, 0.36602063833951404], [0.24812030075187969, 0.291522687103458]], [[0.21654135338345865, 0.3591457825704002], [-0.3157894736842105, 0.17499413541563488], [-0.2992481203007519, 0.1999404529048871]], [[0.38947368421052625, 0.08961904769837144], [-0.02857142857142857, 0.9048224147367736], [-0.009022556390977442, 0.9698849998123449]], [[0.19849624060150375, 0.4015036070038971], [-0.10827067669172932, 0.6495672222012883], [-0.24060150375939846, 0.3068610916782761]], [[-0.09172932330827067, 0.7005157258604466], [-0.12932330827067667, 0.586852724489982], [0.38796992481203, 0.0909728657975423]], [[0.6090225563909774, 0.004370507553662533], [0.4676691729323308, 0.03758851020414889], [0.21954887218045113, 0.352345677311515]], [[0.3714285714285714, 0.10687070696592833], [0.03308270676691729, 0.889875964885619], [0.4195488721804511, 0.065548285182185]], [[0.037593984962406006, 0.8749652018566716], [0.02706766917293233, 0.9098116651774512], [-0.09624060150375939, 0.6864870716570238]], [[0.3157894736842105, 0.17499413541563488], [-0.41503759398496237, 0.06880769728594098], [-0.3609022556390977, 0.11798037156435756]], [[0.021052631578947364, 0.9297994024457079], [0.2902255639097744, 0.21449998226074232], [0.11879699248120301, 0.6178918280262935]], [[-0.12481203007518797, 0.6000744184144979], [0.2150375939849624, 0.36257388702455096], [0.16842105263157892, 0.4778286757964305]], [[0.045112781954887216, 0.8502061902084872], [0.025563909774436087, 0.9148041683144317], [-0.11578947368421053, 0.6268789936728247]], [[0.2721804511278195, 0.24566887828037737], [0.3609022556390977, 0.11798037156435756], [0.045112781954887216, 0.8502061902084872]], [[-0.04661654135338346, 0.8452698052894374], [-0.03458646616541353, 0.8849015088305833], [0.16992481203007515, 0.47384906489411216]], [[-0.08872180451127819, 0.7099206278927328], [0.05563909774436089, 0.8157739948259723], [-0.2796992481203007, 0.23234786067734586]], [[-0.2706766917293233, 0.24839056913669727], [0.4120300751879699, 0.0710472186641245], [-0.31879699248120297, 0.1706978721400242]], [[0.2541353383458646, 0.2795976355152773], [0.13984962406015036, 0.5564935505662203], [0.06616541353383458, 0.7816639121085741]], [[0.19548872180451127, 0.40881908952023194], [-0.5954887218045113, 0.005601441030074218], [-0.08120300751879699, 0.733607630478017]], [[0.1744360902255639, 0.4620108862392819], [0.05714285714285714, 0.8108800461568323], [-0.3654135338345864, 0.11312213950560147]], [[0.03609022556390977, 0.8799311955207728], [0.18045112781954886, 0.4464646102705483], [-0.025563909774436087, 0.9148041683144317]], [[0.22556390977443608, 0.33897093917086685], [0.2781954887218045, 0.23497379059889356], [-0.0631578947368421, 0.7913733978200618]], [[0.03609022556390977, 0.8799311955207728], [-0.28872180451127816, 0.21699266668730374], [-0.07368421052631578, 0.757528699864343]], [[-0.010526315789473682, 0.9648685868194993], [0.13082706766917293, 0.5824730901969897], [0.16691729323308271, 0.48182491843154673]], [[-0.2796992481203007, 0.23234786067734586], [0.060150375939849614, 0.8011125555760907], [0.007518796992481202, 0.9749025019933399]], [[0.08270676691729321, 0.7288508763559796], [-0.048120300751879695, 0.8403389659546905], [-0.43909774436090226, 0.052750717433839675]], [[0.1789473684210526, 0.45032544160933996], [0.08270676691729321, 0.7288508763559796], [-0.11278195488721804, 0.6359170579042353]], [[0.03308270676691729, 0.889875964885619], [-0.26917293233082706, 0.2511314536907905], [-0.07518796992481201, 0.7527267095067718]], [[-0.15037593984962402, 0.5268549482529812], [0.3383458646616541, 0.14452609966359914], [0.06766917293233082, 0.7768207117441497]], [[0.6165413533834586, 0.003789555588157599], [0.18496240601503755, 0.4349861436228526], [0.007518796992481202, 0.9749025019933399]], [[0.5458646616541353, 0.012783628876204734], [-0.5924812030075187, 0.005910465902184196], [-0.19248120300751878, 0.4162064734197989]], [[0.5804511278195488, 0.0072900732244842075], [0.06766917293233082, 0.7768207117441497], [0.12330827067669171, 0.6045088306268851]], [[0.4601503759398496, 0.041200717801954466], [-0.15037593984962402, 0.5268549482529812], [-0.05864661654135337, 0.8059928433862233]], [[0.38646616541353385, 0.09234164249328174], [-0.11879699248120301, 0.6178918280262935], [-0.22857142857142856, 0.3323968570405118]]]
cor1=[i[0][0] for i in correlation_results]
cor2=[i[1][0] for i in correlation_results]
cor3=[i[2][0] for i in correlation_results]

p1=[i[0][1] for i in correlation_results]
p2=[i[1][1] for i in correlation_results]
p3=[i[2][1] for i in correlation_results]

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
df = pd.DataFrame({'x': range(1, len(correlation_results)+1), 'cor1': p1, 'cor2': p2, 'cor3': p3, 'cor1_avg':sum(p1) / float(len(p1)), 'cor2_avg':sum(p2) / float(len(p2)), 'cor3_avg':sum(p3) / float(len(p3))})
print(df)
# multiple line plot
#plt.plot('x', 'cor1', data=df, marker='', color='blue', linewidth=2, label="MF vs KNN")
#plt.plot('x', 'cor2', data=df, marker='', color='red', linewidth=2, label="MF vs BPR")
#plt.plot('x', 'cor3', data=df, marker='', color='green', linewidth=2, label="KNN vs BPR")
plt.plot('x', 'cor1_avg', data=df, marker='', color='blue', linewidth=2, linestyle='dashed', label="MF vs KNN")
plt.plot('x', 'cor2_avg', data=df, marker='', color='red', linewidth=2, linestyle='dashed', label="MF vs BPR")
plt.plot('x', 'cor3_avg', data=df, marker='', color='green', linewidth=2, linestyle='dashed', label="KNN vs BPR")
plt.legend()
plt.show()