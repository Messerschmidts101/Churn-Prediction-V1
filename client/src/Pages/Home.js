import React from 'react'
import Button from '../Components/Button'
import Header1 from '../Components/Header1'
import Footer from '../Components/Footer'

function Home() {

    return (
        <>
            <main>
                <div className='container flex-column'>
                    <Header1>Welcome!</Header1>
                    <Button theme={'primary'}>Get Started!</Button>
                </div>
            </main>
            <Footer />
        </>
    )
}

export default Home