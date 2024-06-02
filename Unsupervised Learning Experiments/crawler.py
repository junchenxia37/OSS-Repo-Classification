import requests
import base64
import time
import json

# class Repo:
#   def __init__(self, name: str, desc: str, readme: str):
#     self.name = name
#     self.desc = desc
#     self.readme = readme

readme_variants = ["README.md", "README", "README.txt", "readme.md", "readme"]

TOKEN = 'xxxxx'

headers = {
    'Authorization': f'token {TOKEN}'
}

def get_readme(repo):
    for readme_filename in readme_variants:
        readme_url = repo['url'] + f"/contents/{readme_filename}"
        response = requests.get(readme_url, headers=headers)
        # time.sleep(0.1)
        if response.status_code == 200:
            readme_data = response.json()
            return base64.b64decode(readme_data['content']).decode()
    return None


def get_repositories(username):
    i = 0
    url = f"https://api.github.com/users/{username}/repos"
    repos = []
    while url:
        response = requests.get(url, headers=headers)
        # time.sleep(0.1)
        
        if response.status_code != 200:
            raise Exception(f"Failed to fetch repos: {response.content}")
        data = response.json()
        for repo in data:
          repo_name = repo['name']
          repo_description = repo['description']
          readme_content = get_readme(repo)
          repos.append({'name': repo['name'], 'desc': repo_description, 'readme': readme_content})
          print(f'{i:<4}{repo_name} done')
          i += 1
        url = response.links.get('next', {}).get('url', None)
    return repos

repos = get_repositories('ethereum')

with open('ethereum_repos', 'w') as f:
   json.dump(repos, f)

# for i, repo in enumerate(repos):
#   desc = repo.desc[:10] if repo.desc is not None else 'NONE'
#   readme = repo.readme[:10] if repo.readme is not None else 'NONE'
#   print(f'{i:<4}{repo.name}\t{desc}\t{readme}')