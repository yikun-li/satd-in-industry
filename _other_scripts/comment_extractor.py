import argparse
import csv
from pathlib import Path

from comment_parser.comment_parser import extract_comments_from_str
from git import Repo

TYPE_MAP = {
    'html': 'text/html',
    'c': 'text/x-c',
    'h': 'text/x-c',
    'cpp': 'text/x-c++',
    'cc': 'text/x-c++',
    'cs': 'text/x-c++',
    'go': 'text/x-go',
    'java': 'text/x-java',
    'js': 'text/x-javascript',
    'py': 'text/x-python',
    'rb': 'text/x-ruby',
    'sh': 'text/x-shellscript',
    'xml': 'text/xml'
}


class Git:
    """
    Git class
    """

    def __init__(self, git_repo_path):
        """
        Initialization

        """
        self._git_repo_path = git_repo_path

    @staticmethod
    def get_diff_simplified(repo, commit):
        """
        Get the differences between changes - simplified version

        :param repo:
        :param commit:
        :return:
        """
        commit_hash = commit.hexsha
        print('Processing commit {}...'.format(commit_hash))
        diff = repo.commit('{}~'.format(commit_hash)).diff(commit_hash, create_patch=True)

        def extract_comments(code, path):
            """
            Extract comments

            """
            if code == '':
                return []

            a_extension = Path(path).suffix[1:].lower()
            if a_extension in TYPE_MAP:
                mime_type = TYPE_MAP[a_extension]
            else:
                mime_type = None

            try:
                if mime_type is None:
                    extracted_comments = []
                else:
                    extracted_comments = extract_comments_from_str(code, mime_type)

                return extracted_comments

            except Exception as e:
                print('{}: {}'.format(e, a_extension))

        def comment_combiner(cs):
            """
            Combine comments

            """
            comments = set()

            if not cs:
                return comments

            previous_is_multiline = True
            previous_line_number = None
            list_comments = []

            for c in cs:
                if c.is_multiline():
                    list_comments.append(c.text())
                    previous_is_multiline = True

                else:
                    if previous_is_multiline:
                        list_comments.append(c.text())
                    else:
                        if (c.line_number() - 1) == previous_line_number:
                            list_comments[-1] = list_comments[-1] + '\n' + c.text()
                        else:
                            list_comments.append(c.text())

                    previous_is_multiline = False
                    previous_line_number = c.line_number()

            return set(list_comments)

        # commit stored in dictionary
        dict_commit = {
            commit.hexsha: {'a_comments': [], 'd_comments': [],
                            'summary': commit.summary, 'message': commit.message,
                            'committer': commit.committer, 'author': commit.author,
                            'parents': [p.hexsha for p in commit.parents],
                            'authored_datetime': commit.authored_datetime,
                            'committed_datetime': commit.committed_datetime}}
        set_all_deleted = set()
        # set_all_added = set()

        for action in ['A', 'R', 'M']:
            for i, changes in enumerate(diff.iter_change_type(action)):
                print('Action: {}-{}'.format(action, i))

                list_changes = changes.diff.decode('utf-8', errors='ignore').splitlines()
                if len(list_changes) == 0:
                    continue

                added, deleted = '', ''
                if action == 'A':
                    added = '\n'.join([line[1:] for line in list_changes if line.startswith('+')])
                elif action == 'D':
                    deleted = '\n'.join([line[1:] for line in list_changes if line.startswith('-')])
                else:
                    added = '\n'.join([line[1:] for line in list_changes if line.startswith('+')])
                    deleted = '\n'.join([line[1:] for line in list_changes if line.startswith('-')])

                # extract comments
                added_comments = extract_comments(added, changes.b_path)
                deleted_comments = extract_comments(deleted, changes.a_path)
                combined_added = comment_combiner(added_comments)
                combined_deleted = comment_combiner(deleted_comments)
                set_all_deleted.update(combined_deleted)
                # set_all_added.update(added_comments)

                for added_comment in combined_added:
                    dict_commit[commit_hash]['a_comments'].append(
                        {'c': added_comment, 'ap': changes.a_path, 'bp': changes.b_path})

        # remove duplicated
        if 'a_comments' in dict_commit[commit_hash]:
            acs = [x for x in dict_commit[commit_hash]['a_comments'] if x['c'] not in set_all_deleted]
            dict_commit[commit_hash]['a_comments'] = acs

        return dict_commit

    def analyze_commits_saving_to_csv(self):
        """
        Analyze all commits and save to DB

        :return:
        """
        print('Analyzing repo in folder {}...'.format(self._git_repo_path))
        git_repo = Repo(self._git_repo_path)
        number_of_all_commits = len(list(git_repo.iter_commits()))
        number_of_commits = git_repo.git.rev_list('--count', 'HEAD')

        with open('comment_extraction.csv', 'a', encoding='utf-8', errors='ignore') as csv_file:
            writer = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
            writer.writerow(['sha', 'summary', 'message', 'author', 'authored_datetime', 'committer',
                             'committed_datetime', 'added_comments'])

            for idx, commit in enumerate(git_repo.iter_commits()):
                print('\nProcessing the {}/{} commit...'.format(idx + 1,
                                                                number_of_all_commits,
                                                                number_of_commits))

                try:
                    c = self.get_diff_simplified(repo=git_repo, commit=commit)
                    dict_c = list(c.values())[0]
                    writer.writerow([list(c.keys())[0], dict_c['summary'], dict_c['message'], dict_c['author'],
                                     dict_c['authored_datetime'], dict_c['committer'], dict_c['committed_datetime'],
                                     dict_c['a_comments']])

                except Exception as e:
                    print(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--git_repo_path", type=str, default="")
    args = parser.parse_args()

    g = Git(args.git_repo_path)
    g.analyze_commits_saving_to_csv()


if __name__ == '__main__':
    main()
