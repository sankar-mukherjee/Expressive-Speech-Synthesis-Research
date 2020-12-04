from sklearn.manifold import TSNE
import numpy as np
import glob
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from bioinfokit.visuz import cluster


def get_speaker_data(spk_embed_path):
    spk_embed_file_names = glob.glob(spk_embed_path + "*.npy")
    spk_embed = []
    spk_label = []
    for f in spk_embed_file_names:
        spk_embed.append(np.load(f))
        # spk_label.append(int(f.replace(spk_embed_path, '').split('_')[0].replace('p', '')))
        spk_label.append(int(f.replace(spk_embed_path, '')[:7].replace('SSB','')))
    spk_embed = np.stack(spk_embed)
    spk_label = np.stack(spk_label)
    return spk_embed, spk_label


if __name__ == "__main__":
    main_dir_path = '/mnt/hdd1/dipjyoti/sankar/Expressive-Speech-Synthesis-Research/'
    # speaker embedding path
    # spk_embed_path = '../../database/ref_audio/for_speaker_tts/speaker_embedding/'
    spk_all_embed_path = '../../database/AISHELL-3/spk_embeds/train/'

    # spk_embed, spk_label = get_speaker_data(spk_embed_path)
    spk_all_embed, spk_all_label = get_speaker_data(spk_all_embed_path)

    # reduce dimension
    model = TSNE(n_components=2, init='pca', perplexity=26.0, n_iter=1000, verbose=1)
    all_em = model.fit_transform(spk_all_embed)
    # test_em = model.fit_transform(spk_embed)


    #cluster.tsneplot(score=all_em, colorlist=spk_label, legendpos='upper right', legendanchor=(1.15, 1))

    cluster.tsneplot(score=all_em)
    np.save('AISHELL_train_speaker_embeddings.npy',all_em)
    #df = pd.DataFrame({'speaker': spk_label, 'embed_x': all_em[:, 0], 'embed_y': all_em[:, 1]})
    #sns_plot = sns.scatterplot(data=df, x="embed_x", y="embed_y", hue="speaker", palette="viridis")
    #plt.show()
    #plt.savefig("output.png")
    print('done')