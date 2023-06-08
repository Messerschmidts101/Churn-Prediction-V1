import React from 'react'
import InputFeature from './InputFeature'
import Label from './Label'

function FeatureList({key}) {
    const feature_names = [
        {
            id: 0,
            name:"state"
        },
        {
            id: 1,
            name: "account_length"
        },
        {
            id: 2,
            name: "area_code"
        },
        {
            id: 3,
            name: "international_plan"
        },
        {
            id: 4,
            name: "voice_mail_plan"
        },
        {
            id: 5,
            name: "number_vmail_messages"
        },
        {
            id: 6,
            name: "total_day_minutes"
        },
        {
            id: 7,
            name: "total_day_calls"
        },
        {
            id: 8,
            name: "total_day_charge"
        },
        {
            id: 9,
            name: "total_eve_minutes"
        },
        {
            id: 10,
            name: "total_eve_calls"
        },
        {
            id: 11,
            name: "total_eve_charge"
        },
        {
            id: 12,
            name: "total_night_minutes"
        },
        {
            id: 13,
            name: "total_night_calls"
        },
        {
            id: 14,
            name: "total_night_charge"
        },
        {
            id: 15,
            name: "total_intl_minutes"
        },
        {
            id: 16,
            name: "total_intl_calls"
        },
        {
            id: 17,
            name: "total_intl_charge"
        },
        {
            id: 18,
            name: "number_customer_service_calls"
        },
    ]
    const feature_list = feature_names.map(feature => {
        <>
            <InputFeature key={feature.id} feature={feature.name} />
            <Label forLabel={feature.name} />
        </>
    })
    return (
        <>
            {feature_list}
        </>
    )
}

export default FeatureList