# silver_bullet_bot/timezone_utils.py (MODIFIED)

import logging
from datetime import datetime, time, timedelta
import pytz

# Import the relaxed window times from config
from silver_bullet_bot.config import SILVER_BULLET_WINDOW_START, SILVER_BULLET_WINDOW_END, MT5_TO_NY_OFFSET
from silver_bullet_bot.config import NY_SESSION_START, NY_SESSION_END
from silver_bullet_bot.config import MT5_TIMEZONE


def setup_timezone_utils():
    """Set up timezone utilities and return logger"""
    logger = logging.getLogger('silver_bullet')
    logger.info("Initializing timezone utilities")
    return logger


def convert_mt5_to_utc(mt5_time):
    """
    Convert MT5 datetime to UTC datetime

    Parameters:
    -----------
    mt5_time : datetime
        Datetime from MT5 (timezone-naive, but actually in MT5_TIMEZONE)

    Returns:
    --------
    datetime
        Datetime in UTC timezone
    """
    if not isinstance(mt5_time, datetime):
        return None

    # MT5 time is timezone-naive but is actually in MT5_TIMEZONE
    mt5_tz = pytz.timezone(MT5_TIMEZONE)

    # Make the time timezone-aware
    if mt5_time.tzinfo is None:
        mt5_time_aware = mt5_tz.localize(mt5_time)
    else:
        mt5_time_aware = mt5_time

    # Convert to UTC
    utc_time = mt5_time_aware.astimezone(pytz.utc)

    return utc_time


def convert_utc_to_ny(utc_time):
    """
    Convert UTC datetime to NY datetime

    Parameters:
    -----------
    utc_time : datetime
        Datetime in UTC (can be timezone-aware or naive)

    Returns:
    --------
    datetime
        Datetime in NY timezone
    """
    if not isinstance(utc_time, datetime):
        return None

    # Make sure the UTC time is timezone-aware
    if utc_time.tzinfo is None:
        utc_time = pytz.utc.localize(utc_time)

    # Convert to NY time
    ny_timezone = pytz.timezone('America/New_York')
    ny_time = utc_time.astimezone(ny_timezone)

    logger = logging.getLogger('silver_bullet')
    logger.debug(f"Time conversion - UTC: {utc_time}, NY: {ny_time}, NY TZ: {ny_time.tzinfo}")

    return ny_time


# silver_bullet_bot/timezone_utils.py (MODIFIED SECTION)

def convert_mt5_to_ny(mt5_time):
    """
    Convert MT5 datetime directly to NY datetime using direct offset

    Parameters:
    -----------
    mt5_time : datetime
        Datetime from MT5 (timezone-naive, assumed to be in UTC+3)

    Returns:
    --------
    datetime
        Datetime in NY timezone
    """
    if not isinstance(mt5_time, datetime):
        return None

    # Direct conversion based on known offsets:
    # MT5 is UTC+3, NY is UTC-4 during EDT (7 hour difference)
    # Create a NY timezone object
    ny_timezone = pytz.timezone('America/New_York')

    # Apply the direct offset (subtract 7 hours during EDT)
    ny_naive_time = mt5_time - timedelta(hours=MT5_TO_NY_OFFSET)

    # Attach the NY timezone
    ny_time = ny_timezone.localize(ny_naive_time)

    # Debug logging
    logger = logging.getLogger('silver_bullet')
    logger.debug(f"Direct timezone conversion: MT5 ({mt5_time}) -> NY ({ny_time})")

    return ny_time


def convert_ny_to_utc(ny_time):
    """
    Convert NY datetime to UTC datetime

    Parameters:
    -----------
    ny_time : datetime
        Datetime in NY timezone

    Returns:
    --------
    datetime
        Datetime in UTC
    """
    if not isinstance(ny_time, datetime):
        return None

    # Make sure the NY time is timezone-aware
    ny_timezone = pytz.timezone('America/New_York')
    if ny_time.tzinfo is None:
        ny_time = ny_timezone.localize(ny_time)

    # Convert to UTC
    utc_time = ny_time.astimezone(pytz.utc)

    return utc_time


def is_ny_trading_time(dt_time):
    """
    Check if the time is within NY trading hours (9:30 AM - 4:00 PM NY time)

    Parameters:
    -----------
    dt_time : datetime
        Datetime (can be in any timezone or naive)

    Returns:
    --------
    bool
        True if within NY trading hours, False otherwise
    """
    # Handle MT5 time if needed
    if dt_time.tzinfo is None:
        # Assume it's MT5 time if timezone-naive
        ny_time = convert_mt5_to_ny(dt_time)
    else:
        # Otherwise convert from whatever timezone it has
        ny_time = dt_time.astimezone(pytz.timezone('America/New_York'))

    # NY trading hours from config
    market_open = NY_SESSION_START
    market_close = NY_SESSION_END

    current_time = ny_time.time()

    # Debug logging
    logger = logging.getLogger('silver_bullet')
    logger.debug(
        f"NY Trading Check - NY Time: {ny_time.strftime('%H:%M:%S')}, Open: {market_open}, Close: {market_close}")

    return market_open <= current_time <= market_close


def is_silver_bullet_window(dt_time):
    """
    Check if the time is within Silver Bullet window (using relaxed window from config)

    Parameters:
    -----------
    dt_time : datetime
        Datetime (can be in any timezone or naive)

    Returns:
    --------
    bool
        True if within Silver Bullet window, False otherwise
    """
    # Handle MT5 time if needed
    if dt_time.tzinfo is None:
        # Assume it's MT5 time if timezone-naive
        ny_time = convert_mt5_to_ny(dt_time)
    else:
        # Otherwise convert from whatever timezone it has
        ny_time = dt_time.astimezone(pytz.timezone('America/New_York'))

    # Print time zone information for debugging
    logger = logging.getLogger('silver_bullet')
    logger.debug(f"NY Time Zone Info: {ny_time.tzinfo}")
    logger.debug(f"NY Time DST Active: {ny_time.dst() != timedelta(0)}")

    # Silver bullet window from config (relaxed)
    window_start = SILVER_BULLET_WINDOW_START
    window_end = SILVER_BULLET_WINDOW_END

    current_time = ny_time.time()

    # Debug logging
    logger.debug(
        f"Silver Bullet Window Check - NY Time: {ny_time.strftime('%H:%M:%S')}, Window: {window_start}-{window_end}")
    logger.debug(
        f"Berlin time should be approximately: {(ny_time + timedelta(hours=6 if ny_time.dst() != timedelta(0) else 5)).strftime('%H:%M:%S')}")

    return window_start <= current_time <= window_end


def get_ny_session_times(utc_day):
    """
    Get NY session start and end times in UTC for a given UTC day

    Parameters:
    -----------
    utc_day : datetime
        A datetime object representing the UTC day

    Returns:
    --------
    tuple
        (session_start, session_end) both in UTC
    """
    # Create NY timezone object
    ny_timezone = pytz.timezone('America/New_York')

    # Convert UTC day to NY day
    ny_day = convert_utc_to_ny(utc_day).replace(hour=0, minute=0, second=0, microsecond=0)

    # Create NY session start and end times using config values
    ny_session_start = ny_timezone.localize(
        datetime.combine(ny_day.date(), NY_SESSION_START)
    )
    ny_session_end = ny_timezone.localize(
        datetime.combine(ny_day.date(), NY_SESSION_END)
    )

    # Convert back to UTC
    utc_session_start = ny_session_start.astimezone(pytz.utc)
    utc_session_end = ny_session_end.astimezone(pytz.utc)

    return utc_session_start, utc_session_end


def get_silver_bullet_window_times(utc_day):
    """
    Get Silver Bullet window start and end times in UTC for a given UTC day

    Parameters:
    -----------
    utc_day : datetime
        A datetime object representing the UTC day

    Returns:
    --------
    tuple
        (window_start, window_end) both in UTC
    """
    # Create NY timezone object
    ny_timezone = pytz.timezone('America/New_York')

    # Convert UTC day to NY day
    ny_day = convert_utc_to_ny(utc_day).replace(hour=0, minute=0, second=0, microsecond=0)

    # Create NY Silver Bullet window start and end times using config values
    ny_window_start = ny_timezone.localize(
        datetime.combine(ny_day.date(), SILVER_BULLET_WINDOW_START)
    )
    ny_window_end = ny_timezone.localize(
        datetime.combine(ny_day.date(), SILVER_BULLET_WINDOW_END)
    )

    # Convert back to UTC
    utc_window_start = ny_window_start.astimezone(pytz.utc)
    utc_window_end = ny_window_end.astimezone(pytz.utc)

    return utc_window_start, utc_window_end


def format_time_for_display(dt):
    """Format datetime for display"""
    if dt.tzinfo is None:
        return dt.strftime('%Y-%m-%d %H:%M:%S (No TZ)')

    utc_time = dt.astimezone(pytz.utc)
    ny_time = dt.astimezone(pytz.timezone('America/New_York'))

    return f"{dt.strftime('%Y-%m-%d %H:%M:%S')} | NY: {ny_time.strftime('%H:%M:%S')} | UTC: {utc_time.strftime('%H:%M:%S')}"


def convert_utc_timestamp_to_ny(utc_timestamp):
    """
    Convert a UTC timestamp to NY time for backtesting purposes

    Parameters:
    -----------
    utc_timestamp : int or datetime
        UTC timestamp (either as epoch seconds or datetime object)

    Returns:
    --------
    datetime
        Datetime in NY timezone
    """
    import pytz
    from datetime import datetime

    # Handle both int timestamps and datetime objects
    if isinstance(utc_timestamp, (int, float)):
        # Convert epoch timestamp to datetime
        utc_dt = datetime.fromtimestamp(utc_timestamp, pytz.UTC)
    elif isinstance(utc_timestamp, datetime):
        # If it's already a datetime, ensure it's UTC-aware
        if utc_timestamp.tzinfo is None:
            utc_dt = pytz.UTC.localize(utc_timestamp)
        else:
            # Convert to UTC if it's in a different timezone
            utc_dt = utc_timestamp.astimezone(pytz.UTC)
    else:
        raise TypeError("timestamp must be int, float, or datetime")

    # Convert to NY time (handles DST automatically)
    ny_timezone = pytz.timezone('America/New_York')
    ny_dt = utc_dt.astimezone(ny_timezone)

    return ny_dt