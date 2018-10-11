# jim-emacs-hack-hylisp-torch 损而损至于列表->知识生长树=>最后组合打出去

### Hylisp function programming list

##### import
```clojure
(import
 torch
 [torch.nn.utils.rnn [pad_sequence]])
```
##### 步步为营保存层,层和模型嫁接迁移
```clojure
(-> (torch.load checkpoint_path :map_location "cpu")
    self.model.load_state_dict)

(self.model encodings embeddings)

```
##### pad_sequence
```clojure
(->
 [(torch.ones 25 300)
  (torch.ones 22 300)
  (torch.ones 15 300)]
 pad_sequence
 (.size)) ;;=> torch.Size([25, 3, 300])

```
##### cat
```clojure
(-> (torch.cat [sen_vecs ctx_vecs] :dim -1) .numpy)
```
##### zeros
```clojure
(torch.zeros (, batch_size max_length feature_dim))
```

