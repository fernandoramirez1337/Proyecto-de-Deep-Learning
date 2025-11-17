# download_data.py
import arxiv
import pandas as pd
from tqdm import tqdm
import time

def download_arxiv_papers():
    papers_data = []
    categories = ['cs.AI', 'cs.LG', 'cs.CV', 'cs.CL']
    papers_per_category = 3000  # 3000 x 4 = 12,000 total
    
    for category in categories:
        print(f"\n Descargando papers de {category}...")
        search = arxiv.Search(
            query=f"cat:{category}",
            max_results=papers_per_category,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        for paper in tqdm(search.results(), total=papers_per_category):
            papers_data.append({
                'id': paper.entry_id,
                'title': paper.title.replace('\n', ' '),
                'abstract': paper.summary.replace('\n', ' '),
                'category': category,
                'all_categories': ','.join(paper.categories),
                'published': paper.published
            })
            time.sleep(0.1)  # Evitar rate limiting
    
    df = pd.DataFrame(papers_data)
    
    # Guardar datos
    df.to_csv('arxiv_papers_raw.csv', index=False)
    print(f"\n Total papers descargados: {len(df)}")
    print(f" Distribucin por categora:")
    print(df['category'].value_counts())
    
    return df

if __name__ == "__main__":
    df = download_arxiv_papers()