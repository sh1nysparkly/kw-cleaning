"""
Keyword Merger & Cleaner
A unified tool for merging keyword data from multiple sources and cleaning/filtering with business-unit-specific rules.
"""

import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from datetime import datetime
import re
from collections import defaultdict

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Keyword Merger & Cleaner",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS & CONFIGURATIONS
# ============================================================================

# Supported data sources
DATA_SOURCES = [
    "Auto-detect",
    "Ahrefs",
    "Conductor",
    "Google Search Console (GSC)",
    "SEMrush",
    "Moz",
    "Other"
]

# Header normalization mapping
HEADER_ALIASES = {
    'query': 'Keyword',
    'top query': 'Keyword',
    'top queries': 'Keyword',
    'search volume': 'Volume',
    'monthly search volume': 'Volume',
    'avg. monthly searches': 'Volume',
    'avg monthly searches': 'Volume',
    'impressions': 'Volume',
    'difficulty': 'KD',
    'keyword difficulty': 'KD',
    'competition': 'Competition',
    'current position': 'Position',
    'position': 'Position',
    'clicks': 'Traffic',
    'current traffic': 'Traffic',
    'current url': 'URL',
    'page': 'URL',
    'top pages': 'URL',
    'page url': 'URL',
    'landing page': 'URL'
}

# Metrics that should be split by source
SOURCE_SPECIFIC_METRICS = ['Volume', 'KD', 'Competition', 'CPC', 'CPS', 'Position', 'Traffic', 'Impressions']

# Metrics that should be aggregated (lists, not source-specific)
AGGREGATED_METRICS = ['SERP Features (All)', 'SERP Features (Owned)', 'Intents']

# Business Unit Negative Keywords
BUSINESS_UNIT_NEGATIVES = {
    "None": "",
    "Insurance": "lawyer|sue|lawsuit|claim",
    "Registries": "girl|signin|login|sign in|log in|portal|www|http|link|adopt|dog|cow|rodeo|vision|cerb|poppy|shoe|barber|chef|gold|sin",
    "Driver Education": "claim|lawyer",
    "Travel (Cruise)": "expedia|costco|tripadvisor|reddit|exoticca|manulife|caasco|amex|american express|travelzoo|sunwing|redtag|selloff|sell off|travelocity|itravel|2000|nagel|transat|collette|river cru|princess cru",
    "Custom": ""
}

# Common negative keyword patterns
COMMON_NEGATIVES = {
    'Competitor/OTA Brand': [
        r"\b(?:expedia|costco|tripadvisor|reddit|exoticca|manulife|caasco|amex|american\s+express|travelzoo|sunwing|redtag|selloff|sell\s*off|travelocity|itravel|2000|nagel|transat|nagel|collette)\b",
    ],
    'Term Misspelling': [
        r"\b(?:carabbean|carabean|carabeen|carabian|caribben|caribean|carribbean|carribean|carrabean)\b",
        r"\b(?:criuse|cruie|cruies|cruis|cruisline|cruiss|cruisw|cruize|cruose|cruse|crusie)\b",
        r"\b(?:pincess|pricess|primcess|princeas|princeness|princeses|princesess|princesss)\b",
        r"\b(?:celbirty|celebarity|celeberty|celebraty|celebrith|celebrityy|celebriy|celerbirty)\b",
        r"\b(?:cunnard|cunrad|curise|curnard|canard)\b",
        r"\b(?:noregian|norewgian|norvegian|norveian|norwe|norwegan|norwegian\s+cr|norwegin|norweigan)\b",
        r"\b(?:oceana|oceanía|oceinia|ceania|ocenia)\b",
    ],
    'Generic/Unwanted Term': [
        r"(?:\s|\.)?(?:com|ca)$",
        r"\b(?:log\s*in|sign\s*in|signon|login|signin|portal|website)\b",
        r"\b(?:address|email|phone|contact\s*service|number)\b",
        r"\b(?:www|career|careers|job|jobs|salary|work|wage|employ|for\s*sale)\b",
        r"\bagent(?:\s+login|\s+portal)?\b",
        r"\b(?:llc|inc|corp|stock|corporation|incorporated|limited|ltd)\b",
        r"\b(?:videos?|picture?)\b",
        r"\b(?:reviews?|forum|youtube|wikipedia|define|instagram|news|newsletter)\b",
        r"\b(?:logo)\b"
    ],
    'Foreign Language': [
        r"\b(?:croisiere|croisieres?|croisière|глобус|jordanie|italie|francais|voyages?|français|prix|vendre|semaines|billet|soir|exotix|suisse|thailande|islande|tout|jours|francophone|angleterre|ecosse|exotique|traditours|circuit|tourisme)\b",
        r"\s+(?:en|de|du|le|au|et|du|les)\s+"
    ],
    'Items/Shopping': [
        r"\b(?:item|accessories?|accessory|magazine|case|jewel|trailer|bag|essentials|gift|luggage|must have)\s+"
    ],
    'Date/Symbol': [
        r"\b20(?:17|18|19|20|21|22|23|24|25)\b",
        r"[.\[\]'√©ßçô&блг®Ä]"
    ]
}

# Location filters
NON_AB_LOCATIONS = [
    r"\bbc\b", r"\bsk\b", r"\bns\b", r"\bmb\b", r"\bnl\b", r"\bnb\b", r"\byt\b", r"\bqc\b", r"\bpq\b",
    "columbia", "bathurst", "afric", "manitoba", "quebec", r"\bus\b", "halifax", "markham",
    "nova scotia", "hst", "pst", "qst", "california", "nwt", "barrie", "brampton", "british",
    "brunswick", "fredericton", "hamilton", "hartford", "kitchener", "michigan", "mile",
    "missisauga", "moncton", "montreal", "montréal", "newfoundland", "ontario", "ottawa",
    "pei", "plano", "prince", "regina", "sask", "yukon", "nova", "spokane", "state",
    "sudbury", "surrey", "delta", "toronto", "troy", r"\btx\b", "vancouver", "waterloo",
    "windsor", "winnipeg", "sturgeon", "kelowna", "whitehorse", "yellowknife", "elmsdale",
    "kamloops", "mirabel", "thunder bay", "victoria", "janvier"
]

ALBERTA_LOCATIONS = [
    r"\bab\b", "alberta", "near", "local", "edmonton", "calgary", "cochrane", "sundre",
    "hinton", "lethbridge", "camrose", "mcmurray", "grande", "prairie", "medicine", "deer",
    "sherwood", r"albert\b", "willow", "jasper", "banff", "kingsway", "sunridge", "manning",
    "crowfoot", "shawnessy", "airdrie", "leduc", "ponoka", "wetask", "hythe", "spruce",
    "okotok", "high river", "acheson", "rocky mountain", "canmore"
]

# Intent patterns
INTENT_PATTERNS = {
    'Informational': r"\b(?:what|where is|where does|why|when|who|how does|how to|which|tip|reddit|guide|tutorial|ideas|example|learn|wiki|question)\b",
    'Commercial': r"\b(?:best|\svs\s|list|compare|review|^top|difference between|alternative|competitor|case study|rating|^rank)\b",
    'Transactional': r"\b(?:affordable|purchase|bargain|cheap|deal|value|buy|shop|coupon|discount|price|pricing|order|sale|cost|how much|estimat|quote|rate|calculat|get|add)\b"
}

# Question keywords
QUESTION_WORDS = ["what", "how", "when", "where", "why", "who", "which", "whose", "will",
                  "would", "should", "could", "can", "do", "does", "did", "are", "is",
                  "was", "were", "have", "vs", "has", "had"]

# Audience patterns
AUDIENCE_PATTERNS = {
    'Couples': r"\b(?:couples?|romantic|romance|for two|partner|anniversar(?:y|ies)|wedding|honeymoon)\b",
    'Solo/Single': r"\b(?:single|solo|alone)\b",
    'Families': r"\b(?:famil(?:y|ies)|kids?|child(?:ren)?|bab(?:y|ies)|teens?|toddlers?|parents?)\b",
    'Seniors': r"\b(?:seniors?|over\s+\d+)\b"
}

# SERP features from Conductor
CONDUCTOR_SERP_COLUMNS = [
    'AI Overview', 'Answer Box', 'Carousel', 'Image', 'Jobs',
    'Local', 'News', 'People Also Ask', 'Product', 'Twitter',
    'Video', 'Video Carousel'
]

# Columns to remove
IRRELEVANT_COLUMNS = ['Jobs', 'News', 'Twitter']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def normalize_header(header):
    """Normalize column headers to standard names."""
    lower_header = str(header).lower().strip()

    # Check aliases
    if lower_header in HEADER_ALIASES:
        return HEADER_ALIASES[lower_header]

    # Handle rank columns (e.g., "October 2025 Rank" -> "Position")
    if lower_header.endswith(' rank'):
        return 'Position'

    # Standardize casing
    if lower_header == 'keyword':
        return 'Keyword'
    if lower_header == 'volume':
        return 'Volume'
    if lower_header == 'kd':
        return 'KD'
    if lower_header == 'cpc':
        return 'CPC'
    if lower_header == 'cps':
        return 'CPS'
    if lower_header == 'competition':
        return 'Competition'
    if lower_header == 'position':
        return 'Position'

    return header


def detect_source_from_filename(filename):
    """Attempt to detect data source from filename."""
    filename_lower = filename.lower()

    if 'ahrefs' in filename_lower or 'ahref' in filename_lower:
        return 'Ahrefs'
    elif 'conductor' in filename_lower:
        return 'Conductor'
    elif 'gsc' in filename_lower or 'search console' in filename_lower or 'google' in filename_lower:
        return 'Google Search Console (GSC)'
    elif 'semrush' in filename_lower or 'sem rush' in filename_lower:
        return 'SEMrush'
    elif 'moz' in filename_lower:
        return 'Moz'
    else:
        return 'Auto-detect'


def parse_uploaded_file(uploaded_file):
    """Parse uploaded CSV, TSV, or Excel file."""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()

        if file_extension in ['xlsx', 'xls', 'xlsm']:
            # Read Excel file
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        elif file_extension in ['csv', 'tsv', 'txt']:
            # Try to detect delimiter
            content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
            delimiter = '\t' if '\t' in content[:1000] else ','
            uploaded_file.seek(0)  # Reset file pointer
            df = pd.read_csv(uploaded_file, delimiter=delimiter, encoding='utf-8', on_bad_lines='skip')
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None

        return df
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        return None


def find_keyword_column(df):
    """Find the keyword column in a dataframe."""
    keyword_headers = ['keyword', 'keywords', 'query', 'top queries', 'top query']

    for col in df.columns:
        if str(col).lower().strip() in keyword_headers:
            return col

    # Check first column
    if str(df.columns[0]).lower().strip() in keyword_headers:
        return df.columns[0]

    return None


def strip_commas_from_numbers(value, metric_name):
    """Remove commas from numeric fields."""
    if pd.isna(value) or value == '':
        return value

    value_str = str(value).replace(',', '').strip()

    # Handle special cases like "< 10" or "low"
    if value_str.lower() in ['< 10', '<10', 'low', 'very low', 'n/a', 'na', '--', '-']:
        return 0

    try:
        # Try to convert to numeric
        return float(value_str) if '.' in value_str else int(value_str)
    except:
        return value


def is_zero_volume(value):
    """Check if a volume value should be considered zero/empty."""
    if pd.isna(value) or value == '':
        return True

    value_str = str(value).strip().lower()

    zero_indicators = ['', '0', '0.0', '0.00', '-', '--', 'n/a', 'na', 'null',
                      'none', 'no data', '< 10', '<10', '<1', 'low', 'very low']

    if value_str in zero_indicators:
        return True

    try:
        num_val = float(value_str)
        return num_val == 0
    except:
        return False


# ============================================================================
# MERGING LOGIC
# ============================================================================

def merge_keyword_data(files_data):
    """
    Merge multiple keyword files with source-specific columns.

    Args:
        files_data: List of tuples [(df, source_name), ...]

    Returns:
        Merged DataFrame with source-specific columns
    """
    keyword_map = defaultdict(dict)
    all_sources = set()
    all_metrics = set()

    for df, source in files_data:
        all_sources.add(source)

        # Normalize headers
        df_normalized = df.copy()
        df_normalized.columns = [normalize_header(col) for col in df_normalized.columns]

        # Find keyword column
        keyword_col = find_keyword_column(df_normalized)
        if not keyword_col:
            st.warning(f"Could not find keyword column in {source} data. Skipping.")
            continue

        # Process each row
        for _, row in df_normalized.iterrows():
            keyword = str(row[keyword_col]).strip()

            if not keyword or keyword == '' or keyword == 'nan':
                continue

            # Initialize keyword entry if new
            if keyword not in keyword_map:
                keyword_map[keyword] = {'Keyword': keyword}

            # Process each column
            for col, value in row.items():
                if col == keyword_col:
                    continue

                # Determine if this is a source-specific metric
                if col in SOURCE_SPECIFIC_METRICS:
                    # Create source-specific column name
                    source_col = f"{col} ({source})"
                    all_metrics.add(col)

                    # Strip commas from numeric values
                    clean_value = strip_commas_from_numbers(value, col)
                    keyword_map[keyword][source_col] = clean_value

                elif col in AGGREGATED_METRICS:
                    # Aggregate lists (like SERP features, intents)
                    if col not in keyword_map[keyword]:
                        keyword_map[keyword][col] = set()

                    if pd.notna(value) and value != '':
                        items = str(value).split(',')
                        for item in items:
                            keyword_map[keyword][col].add(item.strip())

                else:
                    # For other columns, use "first non-empty wins" strategy
                    if col not in keyword_map[keyword] or pd.isna(keyword_map[keyword].get(col)) or keyword_map[keyword].get(col) == '':
                        keyword_map[keyword][col] = value

    # Convert to DataFrame
    rows = []
    for keyword_data in keyword_map.values():
        # Convert sets to comma-separated strings
        for key, value in keyword_data.items():
            if isinstance(value, set):
                keyword_data[key] = ', '.join(sorted(value))
        rows.append(keyword_data)

    merged_df = pd.DataFrame(rows)

    # Sort columns intelligently
    # Priority: Keyword, then source-specific metrics (grouped by metric), then everything else

    ordered_columns = ['Keyword']

    # Add source-specific columns grouped by metric
    for metric in sorted(all_metrics):
        for source in sorted(all_sources):
            col_name = f"{metric} ({source})"
            if col_name in merged_df.columns:
                ordered_columns.append(col_name)

    # Add remaining columns
    for col in merged_df.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)

    # Reorder
    merged_df = merged_df[[col for col in ordered_columns if col in merged_df.columns]]

    return merged_df


# ============================================================================
# CLEANING/FILTERING LOGIC
# ============================================================================

def check_pattern_groups(keyword, pattern_groups):
    """Check if keyword matches any pattern in the groups."""
    for reason, patterns in pattern_groups.items():
        combined_pattern = "(?:" + "|".join(patterns) + ")"
        if re.search(combined_pattern, keyword, re.IGNORECASE):
            return reason
    return None


def create_negative_keyword_pattern(negative_kw_string):
    """Create regex pattern from comma-separated negative keywords."""
    if not negative_kw_string or not isinstance(negative_kw_string, str) or negative_kw_string.strip() == '':
        return None

    try:
        terms = [kw.strip() for kw in negative_kw_string.split(',') if kw.strip()]
        if not terms:
            return None

        # Escape special regex characters
        escaped_terms = [re.escape(term) for term in terms]
        pattern = r"\b(" + "|".join(escaped_terms) + r")\b"
        return re.compile(pattern, re.IGNORECASE)
    except Exception as e:
        st.warning(f"Error creating negative keyword pattern: {e}")
        return None


def create_location_pattern(location_list):
    """Create regex pattern from location list."""
    pattern = "(?:" + "|".join(location_list) + ")"
    return re.compile(pattern, re.IGNORECASE)


def create_question_pattern():
    """Create pattern to detect questions."""
    pattern = r"\b(?:" + "|".join(QUESTION_WORDS) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def find_matching_brand(keyword, brand_names):
    """Find if any brand name matches the keyword."""
    if not brand_names or not isinstance(brand_names, str) or brand_names.strip() == '':
        return ''

    try:
        brands = [b.strip() for b in brand_names.split(',') if b.strip()]

        for brand in brands:
            pattern = r"\b" + re.escape(brand) + r"\b"
            if re.search(pattern, keyword, re.IGNORECASE):
                return brand

        return ''
    except Exception as e:
        return ''


def determine_intent(keyword):
    """Determine search intent based on keyword content."""
    intents = []

    for intent_type, pattern in INTENT_PATTERNS.items():
        if re.search(pattern, keyword, re.IGNORECASE):
            intents.append(intent_type)

    return ', '.join(intents) if intents else ''


def determine_audience(keyword):
    """Determine target audience based on keyword content."""
    for audience_type, pattern in AUDIENCE_PATTERNS.items():
        if re.search(pattern, keyword, re.IGNORECASE):
            return audience_type

    return ''


def clean_keywords(df, config):
    """
    Clean and filter keywords based on configuration.

    Args:
        df: DataFrame with merged keyword data
        config: Dictionary with cleaning configuration

    Returns:
        Tuple of (clean_df, culled_df, stats)
    """
    # Create patterns
    location_exclude_pattern = create_location_pattern(NON_AB_LOCATIONS) if config.get('filter_non_ab_locations') else None
    location_alberta_pattern = create_location_pattern(ALBERTA_LOCATIONS)
    question_pattern = create_question_pattern()

    # Business unit negatives
    bu_negative_pattern = None
    if config.get('business_unit') and config['business_unit'] != 'None':
        bu_negatives = BUSINESS_UNIT_NEGATIVES.get(config['business_unit'], '')
        if bu_negatives:
            bu_negative_pattern = re.compile(bu_negatives, re.IGNORECASE)

    # Custom negatives
    custom_negative_pattern = create_negative_keyword_pattern(config.get('custom_negatives', ''))

    # Brand protection
    brand_names = config.get('brand_names', '')

    # Initialize result containers
    clean_rows = []
    culled_rows = []
    stats = {
        'total': len(df),
        'clean': 0,
        'culled': 0,
        'zero_volume': 0,
        'pattern_match': 0,
        'location': 0,
        'bu_negative': 0,
        'custom_negative': 0
    }

    # Process each row
    for idx, row in df.iterrows():
        keyword = str(row.get('Keyword', '')).strip()

        if not keyword or keyword == '' or keyword == 'nan':
            continue

        should_cull = False
        cull_reason = ''

        # Check for zero volume (if enabled and volume columns exist)
        if config.get('remove_zero_volume'):
            volume_cols = [col for col in df.columns if 'Volume' in col and '(' in col]
            if volume_cols:
                all_zero = all(is_zero_volume(row.get(col)) for col in volume_cols)
                if all_zero:
                    should_cull = True
                    cull_reason = 'Zero Volume'
                    stats['zero_volume'] += 1

        # Check business unit negatives
        if not should_cull and bu_negative_pattern:
            if bu_negative_pattern.search(keyword):
                should_cull = True
                cull_reason = f'Business Unit Negative ({config["business_unit"]})'
                stats['bu_negative'] += 1

        # Check custom negatives
        if not should_cull and custom_negative_pattern:
            if custom_negative_pattern.search(keyword):
                should_cull = True
                cull_reason = 'Custom Negative Keyword'
                stats['custom_negative'] += 1

        # Check location filters
        if not should_cull and location_exclude_pattern:
            has_ab_location = location_alberta_pattern.search(keyword)
            has_non_ab_location = location_exclude_pattern.search(keyword)

            if has_non_ab_location and not has_ab_location:
                should_cull = True
                cull_reason = 'Non-AB Location'
                stats['location'] += 1

        # Check common negative patterns
        if not should_cull and config.get('apply_common_negatives'):
            # First check if brand is protected
            matching_brand = find_matching_brand(keyword, brand_names)
            if not matching_brand:  # Only apply if not a protected brand
                pattern_reason = check_pattern_groups(keyword, COMMON_NEGATIVES)
                if pattern_reason:
                    should_cull = True
                    cull_reason = pattern_reason
                    stats['pattern_match'] += 1

        # Add classification columns
        enhanced_row = row.copy()

        if config.get('classify_questions'):
            enhanced_row['Question'] = 'TRUE' if question_pattern.search(keyword) else 'FALSE'

        if config.get('classify_intent'):
            enhanced_row['Intent'] = determine_intent(keyword)

        if config.get('classify_audience'):
            enhanced_row['Audience'] = determine_audience(keyword)

        if config.get('detect_ab_location'):
            has_ab = location_alberta_pattern.search(keyword)
            enhanced_row['Alberta Location'] = 'TRUE' if has_ab else 'FALSE'

        if brand_names:
            matching_brand = find_matching_brand(keyword, brand_names)
            enhanced_row['Brand Match'] = matching_brand

        # Add to appropriate list
        if should_cull:
            enhanced_row['Cull Reason'] = cull_reason
            culled_rows.append(enhanced_row)
            stats['culled'] += 1
        else:
            clean_rows.append(enhanced_row)
            stats['clean'] += 1

    clean_df = pd.DataFrame(clean_rows) if clean_rows else pd.DataFrame()
    culled_df = pd.DataFrame(culled_rows) if culled_rows else pd.DataFrame()

    return clean_df, culled_df, stats


# ============================================================================
# EXCEL EXPORT
# ============================================================================

def create_excel_output(clean_df, culled_df, report_info):
    """Create multi-tab Excel file with clean data, culled data, and report info."""
    output = BytesIO()

    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        # Cleaned Data tab
        if not clean_df.empty:
            clean_df.to_excel(writer, sheet_name='Cleaned Data', index=False)

        # Culled Rows tab
        if not culled_df.empty:
            culled_df.to_excel(writer, sheet_name='Culled Rows', index=False)

        # Report Info tab
        report_df = pd.DataFrame(report_info.items(), columns=['Setting', 'Value'])
        report_df.to_excel(writer, sheet_name='Report Info', index=False)

    output.seek(0)
    return output


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    st.title("🔍 Keyword Merger & Cleaner")
    st.markdown("Merge keyword data from multiple sources and apply intelligent cleaning filters.")

    # Sidebar for navigation
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode:", ["Merge Keywords", "Clean Keywords", "Merge & Clean (All-in-One)"])

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info(
        "This tool helps you:\n"
        "- Merge keyword data from multiple sources\n"
        "- Create source-specific columns for metrics\n"
        "- Apply business-unit-specific filters\n"
        "- Clean and classify keywords\n"
        "- Export multi-tab Excel reports"
    )

    # ========================================================================
    # MODE 1: MERGE KEYWORDS
    # ========================================================================

    if mode == "Merge Keywords":
        st.header("📊 Merge Keyword Data")
        st.markdown("Upload multiple keyword export files and merge them with source-specific columns.")

        # File uploads
        uploaded_files = st.file_uploader(
            "Upload keyword files (CSV, TSV, or Excel)",
            type=['csv', 'tsv', 'txt', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload files from Ahrefs, Conductor, GSC, SEMrush, etc."
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) uploaded**")

            # Create source selection for each file
            files_data = []

            for i, uploaded_file in enumerate(uploaded_files):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.text(f"📄 {uploaded_file.name}")

                with col2:
                    detected_source = detect_source_from_filename(uploaded_file.name)
                    source = st.selectbox(
                        f"Source for file {i+1}",
                        DATA_SOURCES,
                        index=DATA_SOURCES.index(detected_source) if detected_source in DATA_SOURCES else 0,
                        key=f"source_{i}",
                        label_visibility="collapsed"
                    )

                    # If auto-detect, try to detect
                    if source == "Auto-detect":
                        source = detected_source if detected_source != "Auto-detect" else uploaded_file.name.split('.')[0]

                # Parse file
                df = parse_uploaded_file(uploaded_file)
                if df is not None:
                    files_data.append((df, source))
                    st.caption(f"✓ Parsed: {len(df)} rows, {len(df.columns)} columns")

            st.markdown("---")

            # Merge button
            if st.button("🔄 Merge Data", type="primary", use_container_width=True):
                if len(files_data) < 2:
                    st.warning("Please upload at least 2 files to merge.")
                else:
                    with st.spinner("Merging keyword data..."):
                        merged_df = merge_keyword_data(files_data)

                    st.success(f"✓ Successfully merged {len(merged_df)} unique keywords!")

                    # Display preview
                    st.subheader("Preview")
                    st.dataframe(merged_df.head(20), use_container_width=True)

                    # Download button
                    excel_output = create_excel_output(merged_df, pd.DataFrame(), {
                        'Date Processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Files Merged': len(files_data),
                        'Sources': ', '.join([source for _, source in files_data]),
                        'Total Keywords': len(merged_df)
                    })

                    st.download_button(
                        label="📥 Download Merged Data (Excel)",
                        data=excel_output,
                        file_name=f"merged_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )

                    # Store in session state for cleaning
                    st.session_state['merged_df'] = merged_df
                    st.info("💡 Tip: Switch to 'Clean Keywords' mode to apply filters to this merged data.")

    # ========================================================================
    # MODE 2: CLEAN KEYWORDS
    # ========================================================================

    elif mode == "Clean Keywords":
        st.header("🧹 Clean & Filter Keywords")
        st.markdown("Apply business-unit-specific filters and classification to your keyword data.")

        # Check if we have merged data in session
        if 'merged_df' in st.session_state:
            st.info("Using previously merged data. Upload a new file to start fresh.")
            use_merged = st.checkbox("Use merged data from previous step", value=True)

            if use_merged:
                df_to_clean = st.session_state['merged_df']
            else:
                uploaded_file = st.file_uploader(
                    "Upload keyword file to clean",
                    type=['csv', 'tsv', 'txt', 'xlsx', 'xls']
                )
                if uploaded_file:
                    df_to_clean = parse_uploaded_file(uploaded_file)
                else:
                    df_to_clean = None
        else:
            uploaded_file = st.file_uploader(
                "Upload keyword file to clean",
                type=['csv', 'tsv', 'txt', 'xlsx', 'xls']
            )
            if uploaded_file:
                df_to_clean = parse_uploaded_file(uploaded_file)
            else:
                df_to_clean = None

        if df_to_clean is not None:
            st.caption(f"Loaded: {len(df_to_clean)} rows, {len(df_to_clean.columns)} columns")

            st.markdown("---")

            # Configuration options
            st.subheader("⚙️ Cleaning Configuration")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Filtering Options**")

                business_unit = st.selectbox(
                    "Business Unit",
                    list(BUSINESS_UNIT_NEGATIVES.keys()),
                    help="Pre-defined negative keyword sets for specific business units"
                )

                custom_negatives = st.text_area(
                    "Custom Negative Keywords (comma-separated)",
                    help="Additional keywords to filter out, separated by commas"
                )

                brand_names = st.text_input(
                    "Brand Names to Protect (comma-separated)",
                    help="Brand keywords that should NOT be filtered out"
                )

                filter_non_ab_locations = st.checkbox(
                    "Remove Non-Alberta Location Keywords",
                    value=False,
                    help="Filter out keywords with non-Alberta geographic indicators"
                )

                remove_zero_volume = st.checkbox(
                    "Remove Zero/Low Volume Keywords",
                    value=True,
                    help="Filter out keywords with no search volume"
                )

                apply_common_negatives = st.checkbox(
                    "Apply Common Negative Patterns",
                    value=True,
                    help="Remove competitors, misspellings, generic terms, etc."
                )

            with col2:
                st.markdown("**Classification Options**")

                classify_questions = st.checkbox(
                    "Identify Questions",
                    value=True,
                    help="Add a column flagging question-based keywords"
                )

                classify_intent = st.checkbox(
                    "Classify Search Intent",
                    value=True,
                    help="Add intent classification (Informational, Commercial, Transactional)"
                )

                classify_audience = st.checkbox(
                    "Detect Target Audience",
                    value=True,
                    help="Identify audience segments (Couples, Families, Solo, Seniors)"
                )

                detect_ab_location = st.checkbox(
                    "Detect Alberta Locations",
                    value=True,
                    help="Flag keywords with Alberta-specific locations"
                )

            st.markdown("---")

            # Clean button
            if st.button("🧹 Clean Data", type="primary", use_container_width=True):
                config = {
                    'business_unit': business_unit,
                    'custom_negatives': custom_negatives,
                    'brand_names': brand_names,
                    'filter_non_ab_locations': filter_non_ab_locations,
                    'remove_zero_volume': remove_zero_volume,
                    'apply_common_negatives': apply_common_negatives,
                    'classify_questions': classify_questions,
                    'classify_intent': classify_intent,
                    'classify_audience': classify_audience,
                    'detect_ab_location': detect_ab_location
                }

                with st.spinner("Cleaning keyword data..."):
                    clean_df, culled_df, stats = clean_keywords(df_to_clean, config)

                # Display results
                st.success(f"✓ Cleaning complete!")

                # Stats
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Rows", stats['total'])

                with col2:
                    st.metric("Clean Rows", stats['clean'])

                with col3:
                    st.metric("Culled Rows", stats['culled'])

                with col4:
                    pct_removed = (stats['culled'] / stats['total'] * 100) if stats['total'] > 0 else 0
                    st.metric("% Removed", f"{pct_removed:.1f}%")

                # Breakdown
                st.markdown("**Culled Breakdown:**")
                breakdown_text = f"""
                - Zero Volume: {stats['zero_volume']}
                - Business Unit Negatives: {stats['bu_negative']}
                - Custom Negatives: {stats['custom_negative']}
                - Non-AB Locations: {stats['location']}
                - Pattern Matches: {stats['pattern_match']}
                """
                st.text(breakdown_text)

                # Preview tabs
                tab1, tab2 = st.tabs(["Cleaned Data", "Culled Rows"])

                with tab1:
                    if not clean_df.empty:
                        st.dataframe(clean_df.head(20), use_container_width=True)
                    else:
                        st.warning("No clean rows after filtering.")

                with tab2:
                    if not culled_df.empty:
                        st.dataframe(culled_df.head(20), use_container_width=True)
                    else:
                        st.info("No rows were culled.")

                # Report info
                report_info = {
                    'Date Processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Business Unit': business_unit,
                    'Custom Negatives': custom_negatives if custom_negatives else 'None',
                    'Brand Names': brand_names if brand_names else 'None',
                    'Filter Non-AB Locations': filter_non_ab_locations,
                    'Remove Zero Volume': remove_zero_volume,
                    'Apply Common Negatives': apply_common_negatives,
                    'Total Rows': stats['total'],
                    'Clean Rows': stats['clean'],
                    'Culled Rows': stats['culled'],
                    'Percentage Removed': f"{pct_removed:.1f}%"
                }

                # Download button
                excel_output = create_excel_output(clean_df, culled_df, report_info)

                st.download_button(
                    label="📥 Download Cleaned Data (Excel)",
                    data=excel_output,
                    file_name=f"cleaned_keywords_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

    # ========================================================================
    # MODE 3: MERGE & CLEAN (ALL-IN-ONE)
    # ========================================================================

    elif mode == "Merge & Clean (All-in-One)":
        st.header("🚀 Merge & Clean (All-in-One)")
        st.markdown("Upload multiple files, merge them, and apply cleaning in one streamlined workflow.")

        # File uploads
        uploaded_files = st.file_uploader(
            "Upload keyword files (CSV, TSV, or Excel)",
            type=['csv', 'tsv', 'txt', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help="Upload files from Ahrefs, Conductor, GSC, SEMrush, etc."
        )

        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) uploaded**")

            # Create source selection for each file
            files_data = []

            for i, uploaded_file in enumerate(uploaded_files):
                col1, col2 = st.columns([2, 1])

                with col1:
                    st.text(f"📄 {uploaded_file.name}")

                with col2:
                    detected_source = detect_source_from_filename(uploaded_file.name)
                    source = st.selectbox(
                        f"Source for file {i+1}",
                        DATA_SOURCES,
                        index=DATA_SOURCES.index(detected_source) if detected_source in DATA_SOURCES else 0,
                        key=f"source_all_{i}",
                        label_visibility="collapsed"
                    )

                    if source == "Auto-detect":
                        source = detected_source if detected_source != "Auto-detect" else uploaded_file.name.split('.')[0]

                # Parse file
                df = parse_uploaded_file(uploaded_file)
                if df is not None:
                    files_data.append((df, source))
                    st.caption(f"✓ Parsed: {len(df)} rows, {len(df.columns)} columns")

            st.markdown("---")

            # Configuration options
            st.subheader("⚙️ Cleaning Configuration")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Filtering Options**")

                business_unit = st.selectbox(
                    "Business Unit",
                    list(BUSINESS_UNIT_NEGATIVES.keys()),
                    help="Pre-defined negative keyword sets for specific business units",
                    key="bu_all"
                )

                custom_negatives = st.text_area(
                    "Custom Negative Keywords (comma-separated)",
                    help="Additional keywords to filter out, separated by commas",
                    key="custom_all"
                )

                brand_names = st.text_input(
                    "Brand Names to Protect (comma-separated)",
                    help="Brand keywords that should NOT be filtered out",
                    key="brands_all"
                )

                filter_non_ab_locations = st.checkbox(
                    "Remove Non-Alberta Location Keywords",
                    value=False,
                    help="Filter out keywords with non-Alberta geographic indicators",
                    key="location_all"
                )

                remove_zero_volume = st.checkbox(
                    "Remove Zero/Low Volume Keywords",
                    value=True,
                    help="Filter out keywords with no search volume",
                    key="volume_all"
                )

                apply_common_negatives = st.checkbox(
                    "Apply Common Negative Patterns",
                    value=True,
                    help="Remove competitors, misspellings, generic terms, etc.",
                    key="common_all"
                )

            with col2:
                st.markdown("**Classification Options**")

                classify_questions = st.checkbox(
                    "Identify Questions",
                    value=True,
                    help="Add a column flagging question-based keywords",
                    key="q_all"
                )

                classify_intent = st.checkbox(
                    "Classify Search Intent",
                    value=True,
                    help="Add intent classification (Informational, Commercial, Transactional)",
                    key="intent_all"
                )

                classify_audience = st.checkbox(
                    "Detect Target Audience",
                    value=True,
                    help="Identify audience segments (Couples, Families, Solo, Seniors)",
                    key="audience_all"
                )

                detect_ab_location = st.checkbox(
                    "Detect Alberta Locations",
                    value=True,
                    help="Flag keywords with Alberta-specific locations",
                    key="ab_all"
                )

            st.markdown("---")

            # Process button
            if st.button("🚀 Merge & Clean", type="primary", use_container_width=True):
                if len(files_data) < 1:
                    st.warning("Please upload at least 1 file.")
                else:
                    # Step 1: Merge
                    with st.spinner("Step 1/2: Merging keyword data..."):
                        merged_df = merge_keyword_data(files_data)

                    st.success(f"✓ Merged {len(merged_df)} unique keywords from {len(files_data)} sources")

                    # Step 2: Clean
                    config = {
                        'business_unit': business_unit,
                        'custom_negatives': custom_negatives,
                        'brand_names': brand_names,
                        'filter_non_ab_locations': filter_non_ab_locations,
                        'remove_zero_volume': remove_zero_volume,
                        'apply_common_negatives': apply_common_negatives,
                        'classify_questions': classify_questions,
                        'classify_intent': classify_intent,
                        'classify_audience': classify_audience,
                        'detect_ab_location': detect_ab_location
                    }

                    with st.spinner("Step 2/2: Cleaning keyword data..."):
                        clean_df, culled_df, stats = clean_keywords(merged_df, config)

                    st.success(f"✓ Cleaning complete!")

                    # Stats
                    col1, col2, col3, col4 = st.columns(4)

                    with col1:
                        st.metric("Total Keywords", stats['total'])

                    with col2:
                        st.metric("Clean Keywords", stats['clean'])

                    with col3:
                        st.metric("Culled Keywords", stats['culled'])

                    with col4:
                        pct_removed = (stats['culled'] / stats['total'] * 100) if stats['total'] > 0 else 0
                        st.metric("% Removed", f"{pct_removed:.1f}%")

                    # Breakdown
                    st.markdown("**Culled Breakdown:**")
                    breakdown_text = f"""
                    - Zero Volume: {stats['zero_volume']}
                    - Business Unit Negatives: {stats['bu_negative']}
                    - Custom Negatives: {stats['custom_negative']}
                    - Non-AB Locations: {stats['location']}
                    - Pattern Matches: {stats['pattern_match']}
                    """
                    st.text(breakdown_text)

                    # Preview tabs
                    tab1, tab2 = st.tabs(["Cleaned Data", "Culled Rows"])

                    with tab1:
                        if not clean_df.empty:
                            st.dataframe(clean_df.head(20), use_container_width=True)
                        else:
                            st.warning("No clean rows after filtering.")

                    with tab2:
                        if not culled_df.empty:
                            st.dataframe(culled_df.head(20), use_container_width=True)
                        else:
                            st.info("No rows were culled.")

                    # Report info
                    report_info = {
                        'Date Processed': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'Files Merged': len(files_data),
                        'Sources': ', '.join([source for _, source in files_data]),
                        'Business Unit': business_unit,
                        'Custom Negatives': custom_negatives if custom_negatives else 'None',
                        'Brand Names': brand_names if brand_names else 'None',
                        'Filter Non-AB Locations': filter_non_ab_locations,
                        'Remove Zero Volume': remove_zero_volume,
                        'Apply Common Negatives': apply_common_negatives,
                        'Total Keywords': stats['total'],
                        'Clean Keywords': stats['clean'],
                        'Culled Keywords': stats['culled'],
                        'Percentage Removed': f"{pct_removed:.1f}%"
                    }

                    # Download button
                    excel_output = create_excel_output(clean_df, culled_df, report_info)

                    st.download_button(
                        label="📥 Download Complete Report (Excel)",
                        data=excel_output,
                        file_name=f"keyword_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        use_container_width=True
                    )


if __name__ == "__main__":
    main()
