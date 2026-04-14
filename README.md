# Iris_Classifier_Pro

## תיאור המטלה

פרויקט זה מממש מכונת סיווג (Classifier) עבור מערך הנתונים המפורסם של פרחי האירוס (Iris Dataset).

### המטרה
סיווג פרחי אירוס ל-**4 קטגוריות** על בסיס מאפייניהם הפיזיים, תוך שימוש בשיטות למידת מכונה מפוקחת (Supervised Learning).

### מבנה הפרויקט

```
Iris_Classifier_Pro/
├── README.md
├── src/          # קוד המקור של המודל
├── data/         # מערכי הנתונים
└── docs/         # מסמכים, גרפים ודוחות
    ├── PRD.md
    ├── PLAN.md
    └── TODO.md
```

### טכנולוגיות
- Python
- scikit-learn / NumPy / Pandas
- Matplotlib / Seaborn

### תוצאות סופיות

| מדד | ערך |
|-----|-----|
| **Test Accuracy** | **96.67%** |
| Final MSE | 0.013951 |
| Train / Test | 120 / 30 (80%/20%) |

### שיפורים שיושמו
1. StandardScaler — נרמול z-score
2. פיצול 2D — קטגוריה 4 לפי petal_length AND petal_width
3. MLP + ReLU — שכבה נסתרת עם 16 נוירונים
4. Cross-Entropy Loss — פונקציית שגיאה מתאימה לסיווג

### תוצרים
- מטריצת מבוכה (Confusion Matrix) בגודל 4x4
- מדד דיוק (Accuracy) — **96.67%**
- גרף התכנסות (Convergence Graph) של Cross-Entropy ו-MSE
