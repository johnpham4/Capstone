# Constraint-based Optimization

## 1. Muc tieu
Trong GeoUni, constraint-based optimization la buoc chuyen DSL hinh hoc thanh toa do 2D kha thi de ve hinh. Thay vi giai he rang buoc dang dong, he thong xem moi rang buoc la mot loss differentiable, sau do toi uu tong loss bang gradient-based optimization.

Y tuong cot loi:
- DSL mo ta quan he hinh hoc o muc ky hieu.
- Optimizer bien doi cac quan he do thanh loss terms.
- Adam cap nhat toa do diem de giam vi pham rang buoc.

## 2. Bieu dien bai toan toi uu
Goi tap diem can tim la:

$$
\mathbf{p} = \{(x_1,y_1), (x_2,y_2), \ldots, (x_m,y_m)\}
$$

Voi moi rang buoc hinh hoc $C_i(\mathbf{p})$, he thong muon $C_i(\mathbf{p}) = 0$ khi rang buoc duoc thoa man. Ham muc tieu:

$$
\mathcal{L}(\mathbf{p}) = \sum_i w_i\, C_i(\mathbf{p})^2
$$

Trong implementation:
- Moi term duoc dang ky qua `register_loss(key, val_fn, weight)`.
- Loss thuc te la `weight * (val_fn() ** 2).mean()`.
- Nhom NDG (non-degenerate) dung `register_ndg(...)` voi penalty ham mu de tranh nghiem suy bien.

## 3. Tu DSL den he rang buoc
Pipeline toi uu:
1. DSL parser doc S-expression thanh tuple.
2. Diagram builder map tuple thanh instructions (`Parameter`, `Assertion`).
3. Optimizer duyet instruction va dang ky loss functions tuong ung.
4. Train de tim toa do toi uu.

Vi du map rang buoc:
- `(perpendicular (segment A B) (segment C D))` -> tich vo huong vector bang 0.
- `(parallel (segment A B) (segment C D))` -> tich co huong vector bang 0.
- `(equal-distance A B C D)` -> `|AB| - |CD| = 0`.
- `(on-circle M O)` -> `|OM| - r = 0`.

## 4. Cac constraint primitives chinh
Optimizer su dung cac phep toan co ban tren torch tensors:
- Khoang cach diem-diem `dist(p1,p2)`.
- Dot product cho vuong goc.
- Cross product/area cho thang hang, song song, NDG.
- Angle cosine cho rang buoc goc.
- Khoang cach diem-duong cho projection, incircle, tangent.

Nhung nhom rang buoc duoc dung thuong xuyen:
- Song song, vuong goc.
- Bang nhau ve do dai, bang nhau ve goc.
- Diem nam tren doan/duong/tron.
- Construction constraints: midpoint, centroid, incenter, circumcenter, orthocenter, bisector, projection.

## 5. Khoi tao hinh hoc (Initializer)
Do bai toan phi loi, diem khoi tao anh huong manh den hoi tu. He thong co bo khoi tao thong minh theo tung loai hinh:
- Triangle: scalene, isosceles, equilateral, right, right-isosceles.
- Quadrilateral: square, rectangle, trapezoid, parallelogram, rhombus.
- Mot so template circle va tam dac biet.

Loi ich:
- Giam nguy co roi vao local minima xau.
- Hoi tu nhanh hon.
- Hinh render de nhin va on dinh hon.

## 6. Thuat toan toi uu trong train()
Vong lap toi uu trong code:
1. Khoi tao `optim.Adam(trainable_vars, lr)`.
2. Moi epoch:
   - Tinh lai toan bo loss terms.
   - Thay NaN/Inf bang finite penalty (`nan_to_num`) de an toan so hoc.
   - Cong tong loss va backprop.
   - Gradient clipping (`clip_grad_norm_`).
   - Cap nhat tham so (`optimizer.step()`).
   - Clamp gia tri tham so trong khoang an toan (`param_abs_max`).
3. Early stopping neu loss < `1e-6`.

Config quan trong:
- `epochs` (mac dinh 1000)
- `learning_rate` (mac dinh 0.01)
- `grad_clip_norm` (mac dinh 5.0)
- `param_abs_max` (mac dinh 1e3)

## 7. Multi-start de tang do ben
`solve(n_tries)` chay toi uu nhieu lan voi seed/initialization khac nhau:
- Moi lan thu cho ra mot nghiem va final loss.
- He thong chon nghiem co loss tot nhat.
- Dung som neu dat nguong `eps`.

Dieu nay giup robust hon voi input kho va bai toan co nhieu cuc tri dia phuong.

## 8. Regularization va NDG
Ngoai rang buoc tu DSL, he thong bo sung:
- Regularization de diem khong troi qua xa goc toa do.
- NDG de tranh:
  - diem trung nhau,
  - canh qua ngan,
  - hinh gan suy bien (dien tich gan 0).

NDG la yeu to quan trong de hinh khong chi dung so hoc ma con dung truc quan khi ve.

## 9. Hau xu ly sau toi uu
Sau khi train, `get_diagram()` thuc hien cac buoc on dinh hinh hoc:
- Dich chuyen hinh ve quanh tam de bo cuc can bang.
- Incircle post-correction cho tiep xuc dep hon.
- Tangent post-correction de han che loi cat nhau do sai so so hoc nho.

Muc tieu la nang chat luong hinh render trong use-case thuc te.

## 10. Do phuc tap va hieu nang
Chi phi xap xi:

$$
\mathcal{O}(\text{epochs} \times \text{num\_loss\_terms} \times n\_tries)
$$

Yeu to anh huong thoi gian chay:
- So diem va so rang buoc trong DSL.
- Do kho rang buoc (circle/tangent/multi-construction).
- So lan multi-start.
- Muc can bang weights giua cac term.

## 11. Diem manh va han che
Diem manh:
- Neural-symbolic tach bach: LLM sinh DSL, optimizer dam bao tinh hinh hoc.
- De mo rong rang buoc moi.
- Co co che on dinh (NDG, clipping, clamp, multi-start).

Han che:
- Bai toan phi loi, khong bao dam global optimum.
- Weights mang tinh heuristic, can tuning theo du lieu.
- Rang buoc mau thuan co the khong hoi tu den nghiem dep.

## 12. Doan viet ngan cho paper
Co the dung doan sau cho phan Method:

"We formulate diagram synthesis as a differentiable constraint optimization problem over 2D point coordinates. Each geometric relation is converted into a soft penalty term, and the total objective is defined as a weighted sum of squared constraint violations. Coordinates are optimized using Adam with gradient clipping, non-finite loss guarding, and parameter clamping for numerical stability. To mitigate non-convexity, we apply geometry-aware initialization templates and multi-start optimization, then select the best solution according to final loss. Additional non-degeneracy regularizers and post-corrections are used to improve visual-geometric consistency in rendered diagrams." 

## 13. Tom tat
Constraint-based optimization la trung tam hinh hoc cua GeoUni:
- Nhap: symbolic DSL constraints.
- Xu ly: differentiable optimization tren toa do diem.
- Xuat: diagram kha thi, on dinh, va de render.

Nho buoc nay, he thong co the chuyen mo ta tu nhien thanh hinh ve hinh hoc co tinh nhat quan cao.
