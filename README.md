# Keyword Merger & Cleaner

A unified Streamlit tool for merging keyword data from multiple sources and applying intelligent cleaning/filtering with business-unit-specific rules.

## Features

### Merge Keywords
- Upload multiple keyword export files (CSV, TSV, Excel)
- Auto-detect source from filename (Ahrefs, Conductor, GSC, SEMrush, etc.)
- Manual source override via dropdown
- **Source-specific columns**: Creates separate columns for each metric per source
  - Example: `Volume (Ahrefs)`, `Volume (Conductor)`, `Volume (GSC)`
- Intelligent column ordering (grouped by metric, then by source)
- Non-destructive merging (preserves all data)

### Clean & Filter Keywords
- **Business Unit Negative Keywords**: Pre-defined filters for:
  - Insurance
  - Registries
  - Driver Education
  - Travel/Cruise
  - Custom
- **Custom Negative Keywords**: Add your own comma-separated terms
- **Brand Protection**: Specify brands that should NOT be filtered out
- **Location Filtering**: Remove non-Alberta geographic keywords
- **Zero Volume Filtering**: Remove keywords with no search data
- **Common Negative Patterns**: Auto-filter:
  - Competitors & OTA brands
  - Misspellings
  - Generic/unwanted terms
  - Foreign language keywords
  - Shopping/item keywords
  - Dates & special characters

### Classification & Enrichment
- **Question Detection**: Flags question-based keywords
- **Search Intent Classification**: Informational, Commercial, Transactional
- **Audience Detection**: Couples, Families, Solo/Single, Seniors
- **Location Detection**: Alberta-specific locations

### Non-Destructive Output
Excel file with 3 tabs:
1. **Cleaned Data**: Keywords that passed all filters
2. **Culled Rows**: Keywords that were filtered out (with reason)
3. **Report Info**: Settings, sources, date, statistics

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run the app:
```bash
streamlit run keyword_tool.py
```

### Workflow Options:

**Option 1: Merge Only**
1. Select "Merge Keywords" mode
2. Upload multiple files
3. Select/confirm source for each file
4. Click "Merge Data"
5. Download merged Excel file

**Option 2: Clean Only**
1. Select "Clean Keywords" mode
2. Upload a single file (or use previously merged data)
3. Configure filtering options
4. Click "Clean Data"
5. Download cleaned Excel file

**Option 3: Merge & Clean (All-in-One)**
1. Select "Merge & Clean (All-in-One)" mode
2. Upload multiple files
3. Select sources
4. Configure cleaning options
5. Click "Merge & Clean"
6. Download complete report

## Data Sources Supported

- Ahrefs (Keyword exports, Site Explorer)
- Conductor
- Google Search Console (GSC)
- SEMrush
- Moz
- Any CSV/TSV/Excel with a "Keyword" or "Query" column

## Configuration

### Business Units
Pre-configured negative keyword sets for different business verticals. Select the relevant business unit to automatically apply appropriate filters.

### Custom Negatives
Add your own comma-separated negative keywords that are specific to your project or campaign.

### Brand Protection
Specify your brand names (comma-separated) to prevent them from being filtered out by misspelling or generic term filters.

### Location Filtering
Toggle to keep only Alberta-relevant keywords or include all locations.

## Technical Details

### Source-Specific Merging
The tool creates separate columns for each metric from each source:
- `Volume (Ahrefs)`: 1100
- `Volume (Conductor)`: 1300
- `Volume (GSC)`: 850

This allows you to:
- Compare data across sources
- Identify discrepancies
- Make informed decisions based on source reliability

### Column Ordering
Columns are intelligently ordered:
1. Keyword (always first)
2. Source-specific metrics (grouped by metric type)
3. Aggregated fields (SERP Features, Intents)
4. Classification fields (Intent, Audience, etc.)
5. Other metadata

### Conflict Resolution
For non-metric columns (URL, Parent Keyword, etc.):
- **First non-empty wins**: The first source with data for that field is used
- **Ahrefs priority for Parent Keyword**: If available, Ahrefs data takes precedence

## Output Structure

### Cleaned Data Sheet
All keywords that passed filters with:
- Original data from all sources
- Classification columns (if enabled)
- Enrichment data

### Culled Rows Sheet
All filtered keywords with:
- All original data
- **Cull Reason** column explaining why it was removed

### Report Info Sheet
Processing metadata:
- Date processed
- Sources merged
- Business unit selected
- Filters applied
- Statistics (total, clean, culled, % removed)

## Tips for Non-Technical Users

1. **Start with Merge & Clean mode** for the fastest workflow
2. **Use Auto-detect** for sources - the tool is smart about filenames
3. **Enable all classification options** to get the most enriched data
4. **Always protect your brand names** to avoid accidental filtering
5. **Check the Culled Rows tab** if you think keywords are missing
6. **Save your settings** by noting which business unit and custom negatives you used

## Advanced Use Cases

### Multi-Organization Setup
If you manage multiple brands/organizations:
1. Create separate negative keyword lists per brand
2. Use the "Custom" business unit
3. Paste in brand-specific negatives

### Competitive Analysis
1. Disable "Remove Zero Volume" to keep all keywords
2. Add competitor brand names to custom negatives
3. Enable intent classification to understand competitor targeting

### Content Planning
1. Enable all classification options
2. Filter by Intent = "Informational"
3. Use Audience detection to segment content topics

## Troubleshooting

**Issue**: Source not auto-detected
- **Solution**: Manually select from dropdown or rename file to include source name

**Issue**: Too many keywords culled
- **Solution**: Check "Culled Rows" tab for reasons, adjust filters, disable "Common Negatives"

**Issue**: Brand keywords being removed
- **Solution**: Add brand names to "Brand Names to Protect" field

**Issue**: Excel file won't download
- **Solution**: Check that you have clean rows remaining after filtering

## Future Enhancements

- Keyword clustering (PolyFuzz integration)
- Spell checking with custom dictionaries
- Language detection
- SERP feature normalization (Conductor format)
- Bulk processing mode
- Saved filter templates
- Custom column ordering

## Support

For issues or feature requests, please document:
1. The mode you were using
2. File formats uploaded
3. Filters/settings applied
4. Expected vs. actual behavior
5. Sample data (if possible)

---

**Built for SEO teams who want to spend less time on data cleanup and more time on strategy.**
