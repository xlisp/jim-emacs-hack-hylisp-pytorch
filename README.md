# jim-emacs-hack-hylisp-torch 损而损至于列表->知识生长树=>最后组合打出去

### Hylisp function programming list

##### import
```clojure
(import
 torch
 [torch.nn.utils.rnn [pad_sequence pack_padded_sequence]])
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

