"""Lexical retrieval, hand-rolled. The other half of step 14's hybrid search.

Dense retrieval finds "authentication" when the question says "protect an API".
It cannot find the literal token ``422``: only two documents in the corpus
contain it, and cosine similarity has no way to care. BM25 is the exact inverse,
which is why step 16 fuses the two rather than choosing between them.

The index is built from the chunks already in Qdrant, through the same
``chunk_from_payload`` dense search uses, so the two retrievers cannot drift onto
different chunk sets. Roughly 200 ms for the 1 484-chunk corpus, dominated by
the scroll; a query costs under 5 ms.
"""

import math
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from functools import lru_cache

from qdrant_client import QdrantClient

from app.core.config import Settings
from app.models.chunks import Chunk, ScoredChunk
from app.retrieval.store import chunk_from_payload

# Term-frequency saturation and length normalisation. The Robertson defaults;
# step 16 sweeps the RRF constant and the candidate depth, not these two.
K1 = 1.5
B = 0.75
SCROLL_BATCH = 256

TOKEN = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    r"""Lowercased runs of ``\w``. Applied to documents and queries alike.

    ``\w`` rather than ``[a-z0-9_]`` because a French question carries accents
    and dropping them would make it unmatchable. No stemming and no stopword
    list: BM25's IDF already drives "the" to near zero, which is what a stoplist
    crudely approximates, and a hand-rolled stemmer is a hundred lines of
    guesswork over a corpus whose vocabulary is mostly identifiers.
    """
    return TOKEN.findall(text.lower())


class BM25Index:
    """An inverted index over chunk text, scored with Okapi BM25.

    Built once per process and queried many times, so construction does the work
    the query would otherwise repeat: term frequencies, document lengths and IDF
    are all precomputed here.
    """

    def __init__(self, chunks: Sequence[Chunk], *, k1: float = K1, b: float = B) -> None:
        if not chunks:
            # An empty index silently returns nothing for every query, which
            # reads downstream as "BM25 does not help" rather than "the
            # collection was empty or misspelled".
            raise ValueError("cannot build a BM25 index over no chunks")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.postings: dict[str, dict[int, int]] = defaultdict(dict)
        self.doc_len: list[int] = []
        for index, chunk in enumerate(self.chunks):
            tokens = tokenize(chunk.text)
            self.doc_len.append(len(tokens))
            for term, freq in Counter(tokens).items():
                self.postings[term][index] = freq
        # `or 1.0` guards a corpus of blank chunks; the division below would
        # otherwise raise ZeroDivisionError on the first query.
        self.avg_len = sum(self.doc_len) / len(self.doc_len) or 1.0
        total = len(self.chunks)
        # Lucene's IDF variant. The textbook log((N - df + 0.5) / (df + 0.5))
        # turns negative once df > N/2, so a term appearing in most documents
        # would *subtract* from their scores — wrong in a way that looks
        # plausible right up until you debug a ranking.
        self.idf = {
            term: math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5))
            for term, posting in self.postings.items()
        }

    def search(
        self,
        query: str,
        top_k: int,
        predicate: Callable[[Chunk], bool] | None = None,
    ) -> list[ScoredChunk]:
        """The ``top_k`` best-matching chunks, best first, BM25 scores as computed.

        Scores accumulate only over documents that appear in a query term's
        postings — never over all 1 484 chunks. A query whose every term is
        unknown returns an empty list rather than the least bad chunk.

        ``predicate`` is how payload filters reach this branch; it is applied
        while accumulating, so a filtered query still returns ``top_k`` results
        rather than ``top_k`` minus the ones that were discarded.
        """
        if not query.strip():
            raise ValueError("query is empty")
        if top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {top_k}")

        scores: dict[int, float] = defaultdict(float)
        # A set, so a word repeated in the question does not count twice.
        for term in set(tokenize(query)):
            posting = self.postings.get(term)
            if posting is None:
                continue
            idf = self.idf[term]
            for index, freq in posting.items():
                if predicate is not None and not predicate(self.chunks[index]):
                    continue
                norm = 1 - self.b + self.b * self.doc_len[index] / self.avg_len
                scores[index] += idf * freq * (self.k1 + 1) / (freq + self.k1 * norm)

        # chunk_id breaks ties: two equally scored chunks must not swap places
        # between two benchmark runs of the same commit.
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))
        return [
            ScoredChunk(chunk=self.chunks[index], score=score, rank=rank)
            for rank, (index, score) in enumerate(ranked[:top_k], start=1)
        ]


def build_index(client: QdrantClient, collection: str) -> BM25Index:
    """Scroll every chunk out of ``collection`` and index it.

    ``with_vectors=False`` because BM25 never reads them and 1 536 floats per
    point is the whole cost of the scroll. The loop keeps every page including
    the first, unlike the loop shape in Qdrant's own documentation.
    """
    chunks: list[Chunk] = []
    offset = None
    while True:
        records, offset = client.scroll(
            collection_name=collection,
            limit=SCROLL_BATCH,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        chunks.extend(chunk_from_payload(record.payload or {}) for record in records)
        if offset is None:
            return BM25Index(chunks)


@lru_cache(maxsize=4)
def _index_for(qdrant_url: str, collection: str) -> BM25Index:
    return build_index(QdrantClient(url=qdrant_url), collection)


def default_index(settings: Settings, collection: str) -> BM25Index:
    """The process-wide index for one collection, built on first use.

    Mirrors ``search.default_embedder``: a benchmark run of 45 questions pays the
    ~200 ms build once. Keyed on the URL string rather than on ``settings``
    because ``Settings`` is not hashable.

    ponytail: rebuilt per process, not persisted. A sqlite-backed index would
    save ~200 ms on a CLI call that already spends 1.5-3.7 s in generation.
    """
    return _index_for(settings.qdrant_url, collection)
