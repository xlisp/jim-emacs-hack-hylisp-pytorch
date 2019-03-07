
# jim-emacs-hack-hylisp-pytorch 损而损至于列表->知识生长树=>最后组合打出去

* hylisp-pytorch-examples 来自于 https://github.com/pytorch/examples

- [jim-emacs-hack-hylisp-torch 损而损至于列表->知识生长树=>最后组合打出去](#jim-emacs-hack-hylisp-torch-%E6%8D%9F%E8%80%8C%E6%8D%9F%E8%87%B3%E4%BA%8E%E5%88%97%E8%A1%A8-%E7%9F%A5%E8%AF%86%E7%94%9F%E9%95%BF%E6%A0%91%E6%9C%80%E5%90%8E%E7%BB%84%E5%90%88%E6%89%93%E5%87%BA%E5%8E%BB)
    - [Hylisp function programming list](#hylisp-function-programming-list)
        - [import](#import)
        - [步步为营保存层,层和模型嫁接迁移](#%E6%AD%A5%E6%AD%A5%E4%B8%BA%E8%90%A5%E4%BF%9D%E5%AD%98%E5%B1%82%E5%B1%82%E5%92%8C%E6%A8%A1%E5%9E%8B%E5%AB%81%E6%8E%A5%E8%BF%81%E7%A7%BB)
        - [pad_sequence & pack_padded_sequence](#pad_sequence--pack_padded_sequence)
        - [cat](#cat)
        - [zeros](#zeros)
        - [Embedding](#embedding)
        - [LSTM](#lstm)
        - [view](#view)
        - [contiguous](#contiguous)
        - [transpose](#transpose)
        - [get_word_vector](#get_word_vector)


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
#### SGD

```python
    # 等同于model.parameters(): 单层抽出使用,白盒化和层嫁接思想
    params = [{'params' : md.parameters()} for md in model.modules()
              if md in [model.conv1, model.conv2, model.mp, model.fc]]
    optimizer = optim.SGD(params, lr=args.lr, momentum=args.momentum) # 优化的算法,params是所有层的参数
    # optim.SGD随机梯度下降可只调整某一层的参数
```
