SELECT
    visitorid AS user_id,
    itemid AS product_id,
    event AS event_type,
    TIMESTAMP_SECONDS(CAST(timestamp / 1000 AS INT64)) AS event_time
FROM {{ source('retailrocket_data', 'events') }}
WHERE event IN ('view', 'addtocart', 'purchase')

