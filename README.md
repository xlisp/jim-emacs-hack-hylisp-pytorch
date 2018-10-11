# jim-emacs-hack-hylisp-torch 损而损至于列表->知识生长树=>最后组合打出去

### Hylisp function programming list

##### import
```clojure
(import
 torch
 [torch.nn.utils.rnn [pad_sequence pack_padded_sequence]]
 [torch.nn :as nn]
 [torch.nn.functional :as F])
```
##### 步步为营保存层,层和模型嫁接迁移
```clojure
(-> (torch.load checkpoint_path :map_location "cpu")
    self.model.load_state_dict)

(self.model encodings embeddings)

```
##### pad_sequence & pack_padded_sequence
```clojure
(->
 [(torch.ones 25 300)
  (torch.ones 22 300)
  (torch.ones 15 300)]
 pad_sequence
 (.size)) ;;=> torch.Size([25, 3, 300])

(->
 [(torch.tensor [1 2 3])
  (torch.tensor [3 4])]
 (pad_sequence :batch_first True) ;;不够的补零
 ;;tensor([[1, 2, 3],
 ;;        [3, 4, 0]])
 ;; (pack_padded_sequence :batch_first True :lengths [3 2])
 ;;=> PackedSequence(data=tensor([1, 3, 2, 4, 3]), batch_sizes=tensor([2, 2, 1]))
 ;;(pack_padded_sequence :batch_first True :lengths [2 3])
 ;;=> ValueError: 'lengths' array has to be sorted in decreasing order
 (pack_padded_sequence :batch_first True :lengths [2 2])
 ;;=> PackedSequence(data=tensor([1, 3, 2, 4]), batch_sizes=tensor([2, 2]))
 )
```
##### cat
```clojure
(-> (torch.cat [sen_vecs ctx_vecs] :dim -1) .numpy)
```
##### zeros
```clojure
(torch.zeros (, batch_size max_length feature_dim))
```
##### Embedding
```clojure
((nn.Embedding dict_size char_embedding_size) sentences)
```
##### LSTM
```clojure
((nn.LSTM 512 hidden_size :bidirectional True) combined_embeddings)
```
##### view
```clojure

```
##### contiguous

```clojure

```
##### transpose
```clojure
(-> sen_vecs (.transpose 0 1))
```
##### get_word_vector
```clojure
(.get_word_vector word_embedder "北京")
;;array([-4.8398e-01,  1.5047e-01,  3.7522e-01, -6.6682e-02,  1.1196e-01,
;;       -4.5433e-02,  2.7913e-01,  3.8122e-01, -2.1209e-02,  3.6776e-01,
;;       ....
;;       -2.4237e-01,  1.6338e-01, -5.1791e-02, -1.1458e-01,  9.9900e-02,
;;       -2.0342e-01], dtype=float32)
;;shape: (256,), from s-exp db

(.get_word_vector char_embedder "我")

```