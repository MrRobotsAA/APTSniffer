class TrieNode:
    def __init__(self):
        self.children = {}
        self.count = 0
        self.tags = {}

class SimpleStringMatcher:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, tag):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.count += 1
            if tag in node.tags:
                node.tags[tag] += 1
            else:
                node.tags[tag] = 1

    def count_occurrences(self, word):
        node = self.root
        for char in word:
            if char in node.children:
                node = node.children[char]
            else:
                return 0, {}
        return node.count, node.tags


if __name__ == '__main__':

    trie = SimpleStringMatcher()
    payloads = [([-186, -198, -198, -198, -198, -186, -186, 216, 209], 'tag1'),
                ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'tag2'),
                ([216, 209], 'tag3')]


    for payload, tag in payloads:
        for i in range(len(payload)):
            trie.insert(payload[i:], tag)

    sample_query = [216, 209]
    count, tags = trie.count_occurrences(sample_query)
    print("Occurrences of [216, 209] in Trie:", count)
    print("Tags distribution of [216, 209] in Trie:", tags)
