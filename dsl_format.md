# DSL Input Format (Geometry S-expression)

Tai lieu nay tong hop format dau vao DSL dang duoc su dung trong project.
Nguon tong hop: `pipeline/domain/prompt_dsl.py`, `src/services/diagram/dsl_parser.py`, `src/services/diagram/diagram_builder.py`.

## 1. Nguyen tac tong quat

- Moi dong la **mot S-expression** doc lap.
- Cu phap co dang: `(operator arg1 arg2 ...)`
- Co the viet nhieu dong, moi dong la mot rang buoc/hinh/khai bao.
- Thu tu khuyen nghi:
  1. Shape (`triangle`, `square`, ...)
  2. Define point (`define`)
  3. Segment/line
  4. Circle
  5. Constraint (`parallel`, `perpendicular`, `angle-*`, `equal-distance`, ...)

Luu y: voi `midpoint` va `projection`, doan tham chieu nen duoc khai bao truoc.

## 2. Khuon mau tong quat

```text
(triangle (A B C))
(define M point (midpoint B C))
(segment A M)
```

Parser doc theo token `(`, `)`, va symbol string.

## 3. Shape formats

### 3.1 Triangle

```text
(triangle (A B C))
(triangle (A B C) (isosceles A))
(triangle (A B C) (right B))
(triangle (A B C) (right_isosceles B))
(triangle (A B C) (equilateral))
```

### 3.2 Tu giac/da giac co ten

```text
(quadrilateral (A B C D))
(square (A B C D))
(rectangle (A B C D))
(trapezoid (A B C D))
(parallelogram (A B C D))
(rhombus (A B C D))
```

## 4. Point definition formats

Khuon tong quat:

```text
(define <P> point <construction>)
```

Cac construction pho bien:

```text
(midpoint B C)
(centroid A B C)
(orthocenter A B C)
(incenter A B C)
(circumcenter A B C)
(projection A (segment B C))
(bisector B A C)
(inter-ll C D A B)
(segment A B)
(line A B)
```

Vi du:

```text
(define O point (circumcenter A B C))
(define H point (projection A (segment B C)))
(define D point (segment A B))
```

## 5. Segment / line formats

```text
(segment A B)
(line A B)
```

## 6. Circle formats

Khuon tong quat:

```text
(circle <center>)
(circle <center> (radius 0.5))
(circle <center> <circle-construction>)
```

Cac circle-construction:

```text
(incircle A B C)
(circumcircle A B C)
```

Vi du:

```text
(define O point (incenter A B C))
(circle O (incircle A B C))

Luu y quan trong:
- `incircle/circumcircle` hien tai chi dung cho tam giac (3 diem).
- Bai tu giac noi tiep duong tron tam X: dung `(circle X)` + cac dong `(on-circle ... X)`.
```

## 7. Constraint formats

### 7.1 Song song / vuong goc

```text
(parallel (segment B C) (segment D E))
(perpendicular (segment A B) (segment C D))
```

### 7.2 Goc

```text
(angle-measure A B C 90)
(angle-equal A B C D E F)
```

### 7.3 Do dai bang nhau

```text
(equal-distance A B C D)
(equal-distance A B 1.0)
```

### 7.4 Thuoc doi tuong hinh hoc

```text
(on-circle P O)
(on-segment M A B)

Khuyen nghi:
- Khi can tao diem nam tren doan AB, dung:
  `(define M point (segment A B))`
- Tranh dung `(on-segment ...)` trong construction cua `define`.
```

### 7.5 Tiep tuyen / duong kinh

```text
(tangent T (circle O) AB)
(diameter A B O)

Neu tiep diem la endpoint (vi du tiep tuyen tai A), co the viet:
`(tangent A (circle O) AB)`
```

## 8. Mapping nhanh tieng Viet -> DSL token

- trung diem -> `midpoint`
- trong tam -> `centroid`
- truc tam -> `orthocenter`
- tam noi tiep -> `incenter`
- duong tron noi tiep -> `incircle`
- tam ngoai tiep -> `circumcenter`
- duong tron ngoai tiep -> `circumcircle`
- hinh chieu / chan duong cao -> `projection`
- phan giac -> `bisector`
- song song -> `parallel`
- vuong goc -> `perpendicular`
- goc bang nhau -> `angle-equal`
- goc = so do -> `angle-measure`

## 9. Rule quan trong de dung format

- Khong define trung ten diem nhieu lan.
- Dinh cua shape (A/B/C/D) khong can define lai, tru khi la diem moi.
- Neu dung `projection`, khong can them rang buoc vuong goc/lies-on tuong duong lap lai.
- `on-segment` dung dang 3 diem: `(on-segment M A B)`.
- Khong dung ky hieu toan hoc truc tiep trong output (`⊥`, `∥`, `∠`, `=`), phai dung token DSL.
- Moi menh de goc nen map ro theo thu tu ky tu: `ABC` -> `(angle-* A B C ...)`.
- Cac doan duoc nhac truc tiep trong de (gia thiet/chung minh) nen co dong `(segment X Y)` tuong ung.
- Sau `(diameter A B O)` nen co `(segment A B)` de hien thi day du.
- Constraint tich, tong, bat dang thuc (`AB.EI = ...`, `AC+CD >= ...`, `AB < AC`) hien chua co toan tu DSL truc tiep; can encode qua quan he hinh hoc co san, tranh thay bang `equal-distance` sai nghia.

## 10. Bo mau hoan chinh

### 10.1 Tam giac vuong + duong tron noi tiep

```text
(triangle (A B C) (right B))
(angle-measure B A C 60)
(angle-measure A C B 30)
(define O point (incenter A B C))
(circle O (incircle A B C))
```

### 10.2 Tam giac co trung diem va duong trung binh

```text
(triangle (A B C))
(define D point (midpoint A B))
(define E point (midpoint A C))
(segment D E)
(segment B C)
(parallel (segment B C) (segment D E))
```

### 10.3 Hinh vuong + duong cheo + tam

```text
(square (A B C D))
(define O point (midpoint A C))
(segment A C)
(segment B D)
```

## 11. Ghi chu parser

- Parser hien tai doc duoc S-expression theo token level, khong ep schema o parser.
- Nghia la format hop le ve ngoac/tu khoa van parse duoc; validation nghiep vu theo ontology/rule duoc xu ly o tang khac.

---

Neu ban muon, co the tao them 1 file `docs/dsl_input_format_cheatsheet.md` (1 trang) chi gom bang cu phap ngan gon de copy nhanh khi viet DSL.
